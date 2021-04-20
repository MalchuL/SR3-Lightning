import itertools
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from torch import optim, autograd

from torch.utils.data import DataLoader
import torch.nn as nn

from data import create_dataset

from models import networks, lr_scheduler
from models.loss import CharbonnierLoss

from registry import registries
from transforms.transform import get_transform, get_cartoon_transform
import torch

class SR3Experiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(SR3Experiment, self).__init__()

        self.hparams = hparams
        self.val_folder = Path('output')
        self.val_folder.mkdir(exist_ok=True, parents=True)


        self.create_model()

        # loss
        loss_type = self.hparams.train.pixel_criterion
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.cri_pix = nn.L1Loss()
        elif loss_type == 'l2':
            self.cri_pix = nn.MSELoss()
        elif loss_type == 'cb':
            self.cri_pix = CharbonnierLoss()
        else:
            raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
        self.l_pix_w = self.hparams.train.pixel_weight



    def get_scheduler(self, optimizer):
        args = {**self.hparams.train.scheduler_params}
        args['optimizer'] = optimizer

        return registries.SCHEDULERS.get_from_params(**args)

    def configure_optimizers(self):
        params = self.netG.parameters()
        optimizer = registries.OPTIMIZERS.get_from_params(**{'params': params, **self.hparams.train.optimizer_params})

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': self.get_scheduler(optimizer), 'interval': 'step'}}


    def create_model(self):
        opt = self.hparams
        self.netG = networks.define_G(opt.network_G)


    def forward(self, input):
        return self.netG(input)



    def training_step_end(self, outputs):

        out = outputs['out']
        if len(out) > 0 and self.global_step % self.hparams.train.img_log_freq == 0:
            print('log image')
            out_image = torch.cat([out[name].clone() for name in ['real', 'fake']], dim=0)
            grid = torchvision.utils.make_grid(out_image, nrow=len(out['real']))

            #grid = grid * 0.5 + 0.5
            grid = torch.clamp(grid, 0.0, 1.0)
            self.logger.experiment.add_image('train_image', grid, self.global_step)


        return outputs


    def training_step(self, batch, batch_idx):

        self.var_L = batch['LQ'] # LQ
        self.real_H = batch['GT'] # GT

        self.fake_H = self.netG(self.var_L)
        if self.loss_type == 'fs':
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H) + self.l_fs_w * self.cri_fs(self.fake_H,
                                                                                                      self.real_H)
        elif self.loss_type == 'grad':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            l_pix = l1 + lg
        elif self.loss_type == 'grad_fs':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            lfs = self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
            l_pix = l1 + lg + lfs
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)



        loss = l_pix
        log = {'loss_D': loss}
        out = {'real': self.real_H,
               'fake': self.fake_H,
               'LQ': self.var_L,
               }
        return {'loss': loss, 'out': out, 'log': log, 'progress_bar': log}



    def validation_step(self, batch, batch_nb):

        real = batch['LQ']
        fake = self(real)

        grid = torchvision.utils.make_grid(torch.cat([batch['GT'], fake], dim=0))
        grid = grid * 0.5 + 0.5

        torchvision.utils.save_image(grid, str(self.val_folder / (str(batch_nb) + '.png')), nrow=1)

        return {}

    def validation_epoch_end(self, outputs):
        return {'val_loss': -self.current_epoch}



    def get_transforms(self, isTrain):
        # !!!ATTENTION Here is use blur=False
        return get_transform(self.params, isTrain), get_cartoon_transform(self.params, True), get_transform(self.params, True, use_blur=False)

    def prepare_data(self):
        val_params = self.hparams.datasets.val
        self.val_dataset = create_dataset(val_params)

        train_params = self.hparams.datasets.train
        self.train_dataset = create_dataset(train_params)





    def val_dataloader(self):
        val_params = self.hparams.datasets.val
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False,
                          num_workers=1)

    def train_dataloader(self):
        train_params = self.hparams.datasets.train
        return DataLoader(self.train_dataset,
                          batch_size=train_params.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=train_params.n_workers)
