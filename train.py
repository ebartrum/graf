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
# matplotlib.use('Agg')

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
from GAN_stability.gan_training.config import (
    load_config, build_optimizers,
)


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

    # Save config file
    save_config(os.path.join(out_dir, 'config.yaml'), config)

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)
    # in case of orthographic projection replace focal length by far-near
    assert(not config['data']['orthographic']), "orthographic not yet supported"
    # if config['data']['orthographic']: #TODO: put back in
    #     hw_ortho = (config['data']['far']-config['data']['near'], config['data']['far']-config['data']['near'])
    #     hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr         # add for building generator
    print(train_dataset, hwfr, render_poses.shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # num_workers=config['training']['nworkers'],
        shuffle=True, pin_memory=True, sampler=None, drop_last=True
    )

    val_dataset = train_dataset
    val_loader = train_loader
    hwfr_val = hwfr

    # Create models
    generator, discriminator = build_models(config)
    print('Generator params: %d' % count_trainable_parameters(generator))
    print('Discriminator params: %d, channels: %d' % (count_trainable_parameters(discriminator), discriminator.nc))
    print(generator.render_kwargs_train['network_fn'])
    print(discriminator)

    # Put models on gpu if needed
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    g_optimizer, d_optimizer = build_optimizers(
        generator, discriminator, config
    )

    # input transform
    img_to_patch = ImgToPatch(generator.ray_sampler, hwfr[:3])

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        **generator.module_dict     # treat NeRF specially
    )
    
    # Get model file
    model_file = config['training']['model_file']
    stats_file = 'stats.p'

    # Logger
    logger = Logger(
        log_dir=path.join(out_dir, 'logs'),
        img_dir=path.join(out_dir, 'imgs'),
        monitoring=config['training']['monitoring'],
        monitoring_dir=path.join(out_dir, 'monitoring')
    )

    # Distributions
    ydist = get_ydist(1, device=device)         # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)                 # Dummy to keep GAN training structure in tact
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Save for tests
    n_test_samples_with_same_shape_code = config['training']['n_test_samples_with_same_shape_code']
    ntest = batch_size
    x_real = get_nsamples(train_loader, ntest)
    ytest = torch.zeros(ntest)
    ztest = zdist.sample((ntest,))
    ptest = torch.stack([generator.sample_pose() for i in range(ntest)])
    if n_test_samples_with_same_shape_code > 0:
        ntest *= n_test_samples_with_same_shape_code
        ytest = ytest.repeat(n_test_samples_with_same_shape_code)
        ptest = ptest.unsqueeze_(1).expand(-1, n_test_samples_with_same_shape_code, -1, -1).flatten(0, 1)       # (ntest x n_same_shape) x 3 x 4

        zdim_shape = config['z_dist']['dim'] - config['z_dist']['dim_appearance']
        # repeat shape code
        zshape = ztest[:, :zdim_shape].unsqueeze(1).expand(-1, n_test_samples_with_same_shape_code, -1).flatten(0, 1)
        zappearance = zdist.sample((ntest,))[:, zdim_shape:]
        ztest = torch.cat([zshape, zappearance], dim=1)

    utils.save_images(x_real, path.join(out_dir, 'real.png'))

    # Test generator
    if config['training']['take_model_average']:
        generator_test = copy.deepcopy(generator)
        # we have to change the pointers of the parameter function in nerf manually
        generator_test.parameters = lambda: generator_test._parameters
        generator_test.named_parameters = lambda: generator_test._named_parameters
        checkpoint_io.register_modules(**{k+'_test': v for k, v in generator_test.module_dict.items()})
    else:
        generator_test = generator

    # Evaluator
    evaluator = Evaluator(fid_every > 0, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device, inception_nsamples=33)

    # Train
    tstart = t0 = time.time()

    # Load checkpoint if it exists
    try:
        load_dict = checkpoint_io.load(model_file)
    except FileNotFoundError:
        it = epoch_idx = -1
        fid_best = float('inf')
        kid_best = float('inf')
    else:
        it = load_dict.get('it', -1)
        epoch_idx = load_dict.get('epoch_idx', -1)
        fid_best = load_dict.get('fid_best', float('inf'))
        kid_best = load_dict.get('kid_best', float('inf'))
        logger.load_stats(stats_file)

    # Reinitialize model average if needed
    if (config['training']['take_model_average']
      and config['training']['model_average_reinit']):
        update_average(generator_test, generator, 0.)

    # Learning rate anneling
    d_lr = d_optimizer.param_groups[0]['lr']
    g_lr = g_optimizer.param_groups[0]['lr']
    g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
    d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)
    # ensure lr is not decreased again
    d_optimizer.param_groups[0]['lr'] = d_lr
    g_optimizer.param_groups[0]['lr'] = g_lr

    # Trainer
    trainer = Trainer(
        generator, discriminator, g_optimizer, d_optimizer,
        use_amp=config['training']['use_amp'],
        gan_type=config['training']['gan_type'],
        reg_type=config['training']['reg_type'],
        reg_param=config['training']['reg_param']
    )

    class GRAF(pl.LightningModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def training_step(self, batch, batch_idx, optimizer_idx):
            it = self.global_step
            x_real = batch

            generator.ray_sampler.iterations = it   # for scale annealing

            # Sample patches for real data
            rgbs = img_to_patch(x_real.to(device))          # N_samples x C

            # Discriminator updates
            z = zdist.sample((batch_size,))
            dloss, reg = trainer.discriminator_trainstep(rgbs, y=y, z=z)
            logger.add('losses', 'discriminator', dloss, it=it)
            logger.add('losses', 'regularizer', reg, it=it)

            # Generators updates
            if config['nerf']['decrease_noise']:
              generator.decrease_nerf_noise(it)

            z = zdist.sample((batch_size,))
            gloss = trainer.generator_trainstep(y=y, z=z)
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])

            # Update learning rate
            g_scheduler.step()
            d_scheduler.step()

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']

            logger.add('learning_rates', 'discriminator', d_lr, it=it)
            logger.add('learning_rates', 'generator', g_lr, it=it)

            # (ii) Sample if necessary
            if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
                print("Creating samples...")
                rgb, depth, acc = evaluator.create_samples(ztest.to(device), poses=ptest)
                logger.add_imgs(rgb, 'rgb', it)
                logger.add_imgs(depth, 'depth', it)
                logger.add_imgs(acc, 'acc', it)
            # (vi) Create video if necessary
            if ((it+1) % config['training']['video_every']) == 0:
                N_samples = 4
                zvid = zdist.sample((N_samples,))

                basename = os.path.join(out_dir, '{}_{:06d}_'.format(os.path.basename(config['expname']), it))
                evaluator.make_video(basename, zvid, render_poses, as_gif=True)

        def configure_optimizers(self):
            return ({'optimizer': d_optimizer, 'lr_scheduler': d_scheduler,
                        'frequency': 1},
                   {'optimizer': g_optimizer, 'lr_scheduler': g_scheduler,
                       'frequency': 1})

        def train_dataloader(self):
            return train_loader

    print('it {}: start with LR:\n\td_lr: {}\tg_lr: {}'.format(it, d_optimizer.param_groups[0]['lr'], g_optimizer.param_groups[0]['lr']))

    model = GRAF(config)
    pl_trainer = pl.Trainer(gpus=1, automatic_optimization=False)
    pl_trainer.fit(model) 
