import torch
import torch.nn as nn
import segmentation_models_pytorch as sm
import torch.nn.functional as F


class UNetUpsampler(nn.Module):
    def __init__(self, in_nc, out_nc, scale=4, backbone_name='resnet34'):
        super().__init__()
        self.scale = scale
        self.out_nc = out_nc
        self.in_nc = in_nc * 2 + 1

        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear')
        self.backbone = sm.Unet(encoder_name=backbone_name, in_channels=self.in_nc, classes=self.out_nc)

    def forward(self, x, yt, sigma):

        x_upsampled = self.upsample(x)
        sigma = torch.ones([x_upsampled.shape[0], 1, x_upsampled.shape[2], x_upsampled.shape[3]]).type_as(x) * sigma
        input = torch.cat([x_upsampled, yt, sigma], dim=1)

        N, C, H, W = input.shape
        new_H = H // 32 * 32
        new_W = W // 32 * 32
        if new_H != H or new_W != W:
            input = input[:, :, :new_H, :new_W]

        out = self.backbone(input)
        if new_H != H or new_W != W:
            out = F.pad(out, (0, W - new_W, 0, H - new_H))
        return out
