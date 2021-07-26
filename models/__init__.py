from functools import partial


def get_backbone(name: str):
    # internal to avoid partially initialized modules
    from models import resnet as rn
    from models import resnext as rxn

    try:
        if name.startswith("resnet"):
            params = rn._NETS[name]
            return partial(rn.ResNet, **params)
        elif name.startswith("resnext"):
            params = rxn._NETS[name]
            return partial(rxn.ResNeXt, **params)
    except KeyError:
        raise ValueError(f"Cannot find model '{name}'")
