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
from graf.config import get_dataset, get_hwfr, build_models, build_generator,\
        save_config, update_config, build_lr_scheduler, compute_render_poses,\
        build_discriminator
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
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

class BaseGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator = instantiate(cfg.generator)
        self.discriminator = instantiate(cfg.discriminator)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return training_step.graf(self, batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        g_optimizer = optim.RMSprop(self.generator.parameters(),
                lr=self.cfg.training.lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(self.discriminator.parameters(),
                lr=self.cfg.training.lr_d, alpha=0.99, eps=1e-8)

        g_scheduler = optim.lr_scheduler.StepLR(
            g_optimizer,
            step_size=self.cfg['training']['lr_anneal_every'],
            gamma=self.cfg['training']['lr_anneal'],
            last_epoch=-1
        )
        d_scheduler = optim.lr_scheduler.StepLR(
            d_optimizer,
            step_size=self.cfg['training']['lr_anneal_every'],
            gamma=self.cfg['training']['lr_anneal'],
            last_epoch=-1
        )

        return ({'optimizer': d_optimizer, 'lr_scheduler': d_scheduler,
                    'frequency': 1},
               {'optimizer': g_optimizer, 'lr_scheduler': g_scheduler,
                   'frequency': 1})

    def train_dataloader(self):
        train_dataset = instantiate(self.cfg.dataset.train)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=self.cfg.data.shuffle,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=self.cfg.data.drop_last)

class GRAF(BaseGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        hwfr = get_hwfr(cfg)
        self.img_to_patch = ImgToPatch(self.generator.ray_sampler,
                hwfr[:3])
        self.reg_param = cfg['training']['reg_param']

@hydra.main(config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    assert(not config['data']['orthographic']), "orthographic not yet supported"
    config['data']['fov'] = float(config['data']['fov'])

    tb_logger = CustomTensorBoardLogger('results',
            name=config['expname'], default_hp_metric=False)
    model = GRAF(config)

    callbacks = [GrafSampleGrid(cfg=config['figure_details'],
        parent_dir=tb_logger.log_dir, pl_module=model, monitor=None),
        GrafVideo(cfg=config['figure_details'],
        parent_dir=tb_logger.log_dir, pl_module=model, monitor=None)]
    pl_trainer = pl.Trainer(gpus=1, callbacks=callbacks, logger=tb_logger)
    pl_trainer.fit(model) 

if __name__ == "__main__":
    train()
