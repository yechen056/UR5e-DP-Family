import os

import torch
import torchvision


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: torchvision weights enum string or "r3m"
    """
    if weights in ("r3m", "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def get_r3m(name, **kwargs):
    del kwargs
    import r3m

    original_home = os.environ.get("HOME")
    r3m_home = os.environ.get("R3M_HOME")
    try:
        if r3m_home:
            os.environ["HOME"] = r3m_home
        r3m.device = "cpu"
        model = r3m.load_r3m(name)
    finally:
        if r3m_home:
            if original_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = original_home
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    return resnet_model.to("cpu")
