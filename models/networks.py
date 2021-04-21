import torch
import models.archs.PAN_arch as PAN_arch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.RCAN_arch as RCAN_arch
import registry.registries as registry
from models.archs.UNet import UNetUpsampler

registry.Model(UNetUpsampler)

# Generator
def define_G(opt_net):
    which_model = opt_net['model']

    # image restoration
    if which_model == 'PAN':
        netG = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'MSRResNet_PA':
        netG = SRResNet_arch.MSRResNet_PA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RCAN_PA':
        netG = RCAN_arch.RCAN_PA(n_resgroups=opt_net['n_resgroups'], n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'], res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'], rgb_range=opt_net['rgb_range'], scale=opt_net['scale'])
    else:
        netG = registry.MODELS.get_from_params(**opt_net)

    return netG
