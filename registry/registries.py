import logging

from .registry import Registry

logger = logging.getLogger(__name__)




def _samplers_loader(r: Registry):
    from torch.utils.data import sampler as s

    factories = {
        k: v
        for k, v in s.__dict__.items()
        if "Sampler" in k and k != "Sampler"
    }
    r.add(**factories)


SAMPLER = Registry("sampler")
SAMPLER.late_add(_samplers_loader)
Sampler = SAMPLER.add

class _GradClipperWrap:
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        self.fn(x, *self.args, **self.kwargs)


def _grad_clip_loader(r: Registry):
    from torch.nn.utils import clip_grad as m

    r.add_from_module(m)


# @TODO: why func? should be renamed
GRAD_CLIPPER = Registry("func", default_meta_factory=_GradClipperWrap)
GRAD_CLIPPER.late_add(_grad_clip_loader)


def _modules_loader(r: Registry):
    import torch.nn.modules

    r.add_from_module(torch.nn.modules)


MODULE = Registry("module")
MODULE.late_add(_modules_loader)
Module = MODULE.add





MODEL = Registry("model")
Model = MODEL.add


def _criterion_loader(r: Registry):
    import torch.nn.modules.loss

    r.add_from_module(torch.nn.modules.loss)


CRITERION = Registry("criterion")
CRITERION.late_add(_criterion_loader)
Criterion = CRITERION.add


def _optimizers_loader(r: Registry):
    import torch.optim

    r.add_from_module(torch.optim)


OPTIMIZER = Registry("optimizer")
OPTIMIZER.late_add(_optimizers_loader)
Optimizer = OPTIMIZER.add


def _schedulers_loader(r: Registry):
    import torch.optim.lr_scheduler

    r.add_from_module(torch.optim.lr_scheduler)


SCHEDULER = Registry("scheduler")
SCHEDULER.late_add(_schedulers_loader)
Scheduler = SCHEDULER.add






# backward compatibility

CRITERIONS = CRITERION

GRAD_CLIPPERS = GRAD_CLIPPER
MODELS = MODEL
MODULES = MODULE
OPTIMIZERS = OPTIMIZER

SAMPLERS = SAMPLER
SCHEDULERS = SCHEDULER



__all__ = [


    "Criterion",
    "CRITERION",
    "CRITERIONS",

    "GRAD_CLIPPER",
    "GRAD_CLIPPERS",
    "Model",
    "MODEL",
    "MODELS",
    "Module",
    "MODULE",
    "MODULES",
    "Optimizer",
    "OPTIMIZER",
    "OPTIMIZERS",

    "Sampler",
    "SAMPLER",
    "SAMPLERS",
    "Scheduler",
    "SCHEDULER",
    "SCHEDULERS",

]
