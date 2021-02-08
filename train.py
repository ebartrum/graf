import pytorch_lightning as pl
import argparse
import os
from os import path
import time
import copy
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
import sys

from graf.gan_training import Trainer, Evaluator
from graf.config import get_dataset, get_hwfr, build_models, build_generator,\
        save_config, update_config, build_lr_scheduler, compute_render_poses,\
        build_discriminator
from graf.utils import count_trainable_parameters, get_nsamples
from graf.transforms import ImgToPatch
from graf.figures import GrafSampleGrid, GrafVideo

from submodules.GAN_stability.gan_training import utils
from submodules.GAN_stability.gan_training.train import update_average, toggle_grad, compute_grad2
from submodules.GAN_stability.gan_training.distributions import get_zdist
from submodules.GAN_stability.gan_training.config import load_config, build_optimizers
from graf.logger import CustomTensorBoardLogger
from graf import training_step
import torch.optim as optim
from torchvision import transforms
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
        opt_disc = instantiate(self.cfg.disc_optimiser,
            self.discriminator.parameters())
        opt_gen = instantiate(self.cfg.gen_optimiser,
            self.generator.parameters())
        d_scheduler = instantiate(self.cfg.disc_scheduler, opt_disc)
        g_scheduler = instantiate(self.cfg.gen_scheduler, opt_gen)
        return ({'optimizer': opt_disc, 'lr_scheduler': d_scheduler,
                    'frequency': 1},
               {'optimizer': opt_gen, 'lr_scheduler': g_scheduler,
                   'frequency': 1})

    def train_dataloader(self):
        train_dataset = instantiate(self.cfg.dataset.train,
                transform=self.transform)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.data.shuffle,
            drop_last=self.cfg.data.drop_last)

class GRAF(BaseGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert(not cfg.data.orthographic), "orthographic not yet supported"
        hwfr = get_hwfr(cfg)
        self.img_to_patch = ImgToPatch(self.generator.ray_sampler,
                hwfr[:3])
        self.reg_param = cfg.loss_weight.reg_param
        self.transform = transforms.Compose([
            transforms.Resize(cfg.data.imsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    tb_logger = CustomTensorBoardLogger('results',
            name=cfg.name, default_hp_metric=False)
    model = instantiate(cfg.lm, cfg)

    callbacks = [instantiate(fig, pl_module=model,
                cfg=cfg.figure_details,
                parent_dir=tb_logger.log_dir)
            for fig in cfg.figures.values()]
    pl_trainer = pl.Trainer(gpus=1, callbacks=callbacks, logger=tb_logger)
    pl_trainer.fit(model) 

if __name__ == "__main__":
    train()
