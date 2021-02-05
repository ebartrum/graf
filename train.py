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
from graf.config import get_dataset, get_hwfr, build_models, save_config, update_config, build_lr_scheduler, compute_render_poses
from graf.utils import count_trainable_parameters, get_nsamples
from graf.transforms import ImgToPatch
from graf.figures import GrafSampleGrid, GrafVideo

from GAN_stability.gan_training import utils
from GAN_stability.gan_training.train import update_average, toggle_grad, compute_grad2
from GAN_stability.gan_training.distributions import get_zdist
from GAN_stability.gan_training.config import load_config, build_optimizers
from graf.logger import CustomTensorBoardLogger
from graf import training_step
import torch.optim as optim

class BaseGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator, self.discriminator = build_models(cfg)
        self.g_optimizer, self.d_optimizer = build_optimizers(
                self.generator, self.discriminator, cfg)

        optimizer = config['training']['optimizer']
        lr_g = config['training']['lr_g']
        lr_d = config['training']['lr_d']
        g_params = self.generator.parameters()
        d_params = self.discriminator.parameters()

        self.g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        self.d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)

        step_size = config['training']['lr_anneal_every']
        if isinstance(step_size, str):
            milestones = [int(m) for m in step_size.split(',')]
            self.g_scheduler = optim.lr_scheduler.MultiStepLR(
                self.g_optimizer,
                milestones=milestones,
                gamma=config['training']['lr_anneal'],
                last_epoch=-1)
            self.d_scheduler = optim.lr_scheduler.MultiStepLR(
                self.d_optimizer,
                milestones=milestones,
                gamma=config['training']['lr_anneal'],
                last_epoch=-1)
        else:
            self.g_scheduler = optim.lr_scheduler.StepLR(
                self.g_optimizer,
                step_size=step_size,
                gamma=config['training']['lr_anneal'],
                last_epoch=-1
            )
            self.d_scheduler = optim.lr_scheduler.StepLR(
                self.g_optimizer,
                step_size=step_size,
                gamma=config['training']['lr_anneal'],
                last_epoch=-1
            )

    def training_step(self, batch, batch_idx, optimizer_idx):
        return training_step.graf(self, batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        return ({'optimizer': self.d_optimizer, 'lr_scheduler': self.d_scheduler,
                    'frequency': 1},
               {'optimizer': self.g_optimizer, 'lr_scheduler': self.g_scheduler,
                   'frequency': 1})

    def train_dataloader(self):
        train_dataset = get_dataset(self.cfg)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True, pin_memory=True, sampler=None, drop_last=True)

class GRAF(BaseGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        hwfr = get_hwfr(cfg)
        self.img_to_patch = ImgToPatch(self.generator.ray_sampler,
                hwfr[:3])
        self.reg_param = cfg['training']['reg_param']

parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.'
)
parser.add_argument('config', type=str, help='Path to config file.')
args, unknown = parser.parse_known_args() 
config = load_config(args.config, 'configs/default.yaml')
config = update_config(config, unknown)
assert(not config['data']['orthographic']), "orthographic not yet supported"
config['data']['fov'] = float(config['data']['fov'])

tb_logger = CustomTensorBoardLogger('results/',
        name=config['expname'], default_hp_metric=False)
model = GRAF(config)
config['figure_details'] = {'dir': os.path.join(tb_logger.log_dir,'figures'),
        'filename': None,
        'ntest': 8,
        'noise_dim': config['z_dist']['dim'],
        'data': config['data']}

callbacks = [GrafSampleGrid(cfg=config['figure_details'],
    parent_dir='.', pl_module=model, monitor=None), GrafVideo(cfg=config['figure_details'],
    parent_dir='.', pl_module=model, monitor=None)]
pl_trainer = pl.Trainer(gpus=1, callbacks=callbacks,
        logger=tb_logger,automatic_optimization=False)
pl_trainer.fit(model) 
