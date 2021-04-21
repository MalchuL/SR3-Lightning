import itertools
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
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
import torch.nn.functional as F

class SR3Experiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(SR3Experiment, self).__init__()

        self.hparams = hparams
        self.val_folder = Path('output')
        self.val_folder.mkdir(exist_ok=True, parents=True)


        self.create_model()
        self.scale = self.hparams.scale

        self.sigma_begin = self.hparams.sigma_begin
        self.sigma_end = self.hparams.sigma_end
        self.num_sigmas = self.hparams.num_sigmas

        self.sigmas = np.exp(
            np.linspace(np.log(self.sigma_begin),
                        np.log(self.sigma_end),
                        self.num_sigmas, dtype=np.float32))

        print('sigmas', self.sigmas)
        # loss
        loss_type = self.hparams.train.pixel_criterion
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.cri_pix = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.cri_pix = nn.MSELoss(reduction='none')
        elif loss_type == 'cb':
            self.cri_pix = CharbonnierLoss()
        elif loss_type == 'l1l2':
            class L1L2Loss(nn.Module):
                def __init__(self, reduction='mean'):
                    super().__init__()
                    self.l1 = nn.L1Loss(reduction=reduction)
                    self.l2 = nn.MSELoss(reduction=reduction)

                def forward(self, x, y):
                    return (self.l1(x, y) + self.l2(x, y)) * 0.5

            self.cri_pix = L1L2Loss(reduction='none')
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


    def forward(self, input, yt, sigmas):
        return self.netG(input, yt, sigmas)



    def training_step_end(self, outputs):

        out = outputs['out']
        if len(out) > 0 and self.global_step % self.hparams.train.img_log_freq == 0:
            print('log image')
            elems = min(out['real'].shape[0], self.hparams.train.img_to_log)
            out_image = torch.cat([out[name].clone()[:elems] for name in ['real', 'fake', 'noisy']], dim=0)
            grid = torchvision.utils.make_grid(out_image, nrow=elems)

            grid = grid * 0.5 + 0.5
            grid = torch.clamp(grid, 0.0, 1.0)
            self.logger.experiment.add_image('train_image', grid, self.global_step)


        return outputs


    def get_prior_image(self, tensor):
        N, C, H, W = tensor.shape
        prior = torch.randn([N, C, H * self.scale, W * self.scale]).type_as(tensor)
        return prior

    def add_noise(self, input, sigma):
        noise = torch.randn_like(input)

        output = torch.sqrt(sigma) * input + torch.sqrt(1 - sigma) * noise
        return output, noise

    def invert_noise(self, output, noise, sigma): # look on add_noise func
       return (output - torch.sqrt(1 - sigma) * noise) / torch.sqrt(sigma)

    def get_sigmas(self, sigmas, batch_size, is_prod=True):
        if is_prod:
            labels = np.ones([batch_size, len(sigmas)])
            teta = np.random.randint(0, len(sigmas), batch_size)
            for i in range(batch_size):
                labels[i,:teta[i]] = sigmas[:teta[i]]
            labels = np.prod(labels, 1)
        else:
            labels = np.random.choice(sigmas, batch_size)
        return labels.reshape([-1, 1, 1, 1])

    def training_step(self, batch, batch_idx):



        input = batch['LQ'] # LQ
        target = batch['GT'] # GT


        sigmas = torch.from_numpy(self.get_sigmas(self.sigmas, input.shape[0])).type_as(input)
        noisy_input, noise = self.add_noise(target, sigmas)

        self.fake_H = self(input, noisy_input, sigmas)
        if self.loss_type == 'fs':
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, target) + self.l_fs_w * self.cri_fs(self.fake_H,
                                                                                                      target)
        elif self.loss_type == 'grad':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, target)
            lg = self.l_grad_w * self.gradloss(self.fake_H, target)
            l_pix = l1 + lg
        elif self.loss_type == 'grad_fs':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, target)
            lg = self.l_grad_w * self.gradloss(self.fake_H, target)
            lfs = self.l_fs_w * self.cri_fs(self.fake_H, target)
            l_pix = l1 + lg + lfs
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, noise)
            l_pix = (l_pix / sigmas).mean()




        loss = l_pix
        log = {'loss': loss}
        out = {'real': target,
               'fake': self.invert_noise(noisy_input, self.fake_H, sigmas),
               'noisy': noisy_input,
               'LQ': input,
               }
        return {'loss': loss, 'out': out, 'log': log, 'progress_bar': log}



    def validation_step(self, batch, batch_nb):

        real = batch['LQ']
        yt = self.get_prior_image(real)
        teta_t = self.sigmas[0]
        eps = 1e-4


        log_yt = []

        SHAPE_GF_INFER = False

        for i in range(self.num_sigmas):

            alpha_t = self.sigmas[-1-i]
            teta_t *= alpha_t


            if SHAPE_GF_INFER:
                step = math.sqrt(2)
                if i < self.num_sigmas - 1:
                    yt = yt - self(real, yt, teta_t)
                    z = self.get_prior_image(real)
                    yt += step * teta_t * z
                else:
                    yt = yt - self(real, yt, teta_t)
            else:
                multiplier = ((1 - alpha_t) / (np.sqrt(1 - teta_t + eps)))
                yt = (1 / np.sqrt(alpha_t)) * (yt - multiplier * self(real, yt, teta_t))
                if i < self.num_sigmas - 1:
                    z = self.get_prior_image(real)
                    yt += np.sqrt(1 - alpha_t) * z

            log_yt.append(yt)

        self.log('val_min', yt.min())
        self.log('val_max', yt.max())

        grid = torchvision.utils.make_grid(torch.cat([batch['GT'], yt], dim=0))
        grid = grid * 0.5 + 0.5
        grid = torch.clamp(grid, 0, 1)

        torchvision.utils.save_image(grid, str(self.val_folder / (str(batch_nb) + '.png')), nrow=1)

        if batch_nb == 0:
            log_yt = log_yt[:min(len(log_yt), self.hparams.train.img_to_log)]
            grid_tensor = torch.cat([batch['GT'], *log_yt], dim=0)
            grid = torchvision.utils.make_grid(grid_tensor, nrow=len(grid_tensor))
            grid = grid * 0.5 + 0.5
            grid = torch.clamp(grid, 0, 1)
            self.logger.experiment.add_image('valid_image', grid, self.current_epoch)

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
