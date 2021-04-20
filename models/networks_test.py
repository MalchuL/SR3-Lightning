import registry.registries as registry
from models.decoders.folding_net_dec import FoldingNetDec, FoldingNetDec3d, FoldingNetDec3dSphere
from models.decoders.point_net_dec import PointNetDecoder
from models.encoders.PointCloudTransformer import PointTransformerFeaturizer
from models.encoders.l3dp_encoder import ShapeGFEncoder
from models.decoders.resnet_add import ShapeGFDecoder
from models.decoders.resnet_cbn import ShapeGFConditionalDecoder
from models.encoders.PointNet2 import PointNet2Featurizer


registry.Model(ShapeGFDecoder)
registry.Model(ShapeGFConditionalDecoder)
registry.Model(ShapeGFEncoder)
registry.Model(PointNet2Featurizer)
registry.Model(PointTransformerFeaturizer)

registry.Model(FoldingNetDec)
registry.Model(PointNetDecoder)
registry.Model(FoldingNetDec3d)
registry.Model(FoldingNetDec3dSphere)

def define_encoder(name, args):
    return registry.MODELS.get_instance(name, **args)

def define_decoder(name, args):
    return registry.MODELS.get_instance(name, **args)


def define_encoder_from_params(args):
    print(registry.MODELS.keys())
    return registry.MODELS.get_from_params(**args)

def define_decoder_from_params(args):
    return registry.MODELS.get_from_params(**args)