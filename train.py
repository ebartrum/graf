import pytorch_lightning as pl
import argparse
import os
from os import path
import time
import copy
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Trainer, Evaluator
from graf.config import get_data, build_models, save_config, update_config, build_lr_scheduler
from graf.utils import count_trainable_parameters, get_nsamples
from graf.transforms import ImgToPatch

from GAN_stability.gan_training import utils
from GAN_stability.gan_training.train import update_average, toggle_grad, compute_grad2
from GAN_stability.gan_training.logger import Logger
from GAN_stability.gan_training.checkpoints import CheckpointIO
from GAN_stability.gan_training.distributions import get_ydist, get_zdist
from GAN_stability.gan_training.config import load_config, build_optimizers

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')

    args, unknown = parser.parse_known_args() 
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    restart_every = config['training']['restart_every']
    fid_every = config['training']['fid_every']
    save_every = config['training']['save_every']
    backup_every = config['training']['backup_every']
    save_best = config['training']['save_best']
    assert save_best=='fid' or save_best=='kid', 'Invalid save best metric!'

    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)
    assert(not config['data']['orthographic']), "orthographic not yet supported"

    config['data']['hwfr'] = hwfr         # add for building generator

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )

    val_dataset = train_dataset
    val_loader = train_loader
    hwfr_val = hwfr

    # Logger
    logger = Logger(
        log_dir=path.join(out_dir, 'logs'),
        img_dir=path.join(out_dir, 'imgs'),
        monitoring=config['training']['monitoring'],
        monitoring_dir=path.join(out_dir, 'monitoring')
    )


    # Train
    tstart = t0 = time.time()

    class GRAF(pl.LightningModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.generator, self.discriminator = build_models(config)
            self.g_optimizer, self.d_optimizer = build_optimizers(
                    self.generator, self.discriminator, cfg)
            self.img_to_patch = ImgToPatch(self.generator.ray_sampler, hwfr[:3])

            self.ydist = get_ydist(1)         # Dummy to keep GAN training structure in tact
            self.y = torch.zeros(batch_size)                 # Dummy to keep GAN training structure in tact
            self.zdist = get_zdist(cfg['z_dist']['type'], cfg['z_dist']['dim'])

            # Save for tests
            n_test_samples_with_same_shape_code = config['training']['n_test_samples_with_same_shape_code']
            ntest = batch_size
            x_real = get_nsamples(train_loader, ntest)
            ytest = torch.zeros(ntest)
            self.ztest = self.zdist.sample((ntest,))
            self.ptest = torch.stack([self.generator.sample_pose() for i in range(ntest)])

            self.generator_test = self.generator
            self.evaluator = Evaluator(fid_every > 0, self.generator_test,
                    self.zdist, self.ydist, batch_size=batch_size,
                    inception_nsamples=33)

            # Learning rate anneling
            d_lr = self.d_optimizer.param_groups[0]['lr']
            g_lr = self.g_optimizer.param_groups[0]['lr']
            self.g_scheduler = build_lr_scheduler(self.g_optimizer, cfg, last_epoch=-1)
            self.d_scheduler = build_lr_scheduler(self.d_optimizer, cfg, last_epoch=-1)
            # ensure lr is not decreased again
            self.d_optimizer.param_groups[0]['lr'] = d_lr
            self.g_optimizer.param_groups[0]['lr'] = g_lr

            self.gan_trainer = Trainer(
                self.generator, self.discriminator, self.g_optimizer,
                self.d_optimizer, use_amp=config['training']['use_amp'],
                gan_type=config['training']['gan_type'],
                reg_type=config['training']['reg_type'],
                reg_param=config['training']['reg_param'])

        def training_step(self, batch, batch_idx, optimizer_idx):
            it = self.global_step
            x_real = batch

            self.generator.ray_sampler.iterations = it   # for scale annealing

            # Sample patches for real data
            rgbs = self.img_to_patch(x_real.to(self.device))          # N_samples x C

            # Discriminator updates
            z = self.zdist.sample((batch_size,))
            dloss, reg = self.gan_trainer.discriminator_trainstep(rgbs,
                    y=self.y, z=z)
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            if config['nerf']['decrease_noise']:
              self.generator.decrease_nerf_noise(it)

            z = self.zdist.sample((batch_size,))
            gloss = self.gan_trainer.generator_trainstep(y=self.y, z=z)
            logger.add('losses', 'generator', gloss, it=it)

            # Update learning rate
            self.g_scheduler.step()
            self.d_scheduler.step()

            d_lr = self.d_optimizer.param_groups[0]['lr']
            g_lr = self.g_optimizer.param_groups[0]['lr']

            logger.add('learning_rates', 'discriminator', d_lr, it=it)
            logger.add('learning_rates', 'generator', g_lr, it=it)

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                print("Creating samples...")
                rgb, depth, acc = self.evaluator.create_samples(
                        self.ztest, poses=self.ptest)
                logger.add_imgs(rgb, 'rgb', it)
                logger.add_imgs(depth, 'depth', it)
                logger.add_imgs(acc, 'acc', it)
            # (vi) Create video if necessary
            if ((it+1) % config['training']['video_every']) == 0:
                N_samples = 4
                zvid = self.zdist.sample((N_samples,))

                basename = os.path.join(out_dir, '{}_{:06d}_'.format(os.path.basename(config['expname']), it))
                self.evaluator.make_video(basename, zvid, render_poses, as_gif=True)

        def configure_optimizers(self):
            return ({'optimizer': self.d_optimizer, 'lr_scheduler': self.d_scheduler,
                        'frequency': 1},
                   {'optimizer': self.g_optimizer, 'lr_scheduler': self.g_scheduler,
                       'frequency': 1})

        def train_dataloader(self):
            return train_loader

    model = GRAF(config)
    pl_trainer = pl.Trainer(gpus=1, automatic_optimization=False)
    pl_trainer.fit(model) 
