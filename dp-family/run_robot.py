if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib


class _TorchXpuCompat:
    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def manual_seed(seed):
        return None

    @staticmethod
    def manual_seed_all(seed):
        return None


if not hasattr(torch, "xpu"):
    torch.xpu = _TorchXpuCompat()

from train import instantiate_workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'config'))
)
def main(cfg):
    workspace = instantiate_workspace(cfg)
    workspace.run_robot()

if __name__ == "__main__":
    main()
