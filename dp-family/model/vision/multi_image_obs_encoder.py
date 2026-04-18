from typing import Dict, Tuple, Union
import copy

import torch
import torch.nn as nn
import torchvision

from common.pytorch_util import replace_submodules
from model.common.module_attr_mixin import ModuleAttrMixin


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
    ):
        super().__init__()

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            attr_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if attr_type == "rgb":
                rgb_keys.append(key)
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=max(1, x.num_features // 16),
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                transforms = []
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    transforms.append(torchvision.transforms.Resize(size=(h, w)))
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    crop_cls = torchvision.transforms.RandomCrop if random_crop else torchvision.transforms.CenterCrop
                    transforms.append(crop_cls(size=(h, w)))
                if imagenet_norm:
                    transforms.append(
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        )
                    )
                key_transform_map[key] = nn.Sequential(*transforms) if transforms else nn.Identity()
            elif attr_type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {attr_type}")

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = []
        if self.share_rgb_model:
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                batch_size = img.shape[0] if batch_size is None else batch_size
                assert batch_size == img.shape[0]
                if img.ndim == 5:
                    batch, steps = img.shape[:2]
                    img = img.reshape(batch * steps, *img.shape[2:])
                if img.shape[1:] != self.key_shape_map[key] and img.shape[-1] == self.key_shape_map[key][0]:
                    img = torch.moveaxis(img, -1, 1)
                assert img.shape[1:] == self.key_shape_map[key]
                imgs.append(self.key_transform_map[key](img))
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["rgb"](imgs)
            feature = feature.reshape(len(self.rgb_keys), -1, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                batch_size = img.shape[0] if batch_size is None else batch_size
                assert batch_size == img.shape[0]
                if img.ndim == 5:
                    batch, steps = img.shape[:2]
                    img = img.reshape(batch * steps, *img.shape[2:])
                if img.shape[1:] != self.key_shape_map[key] and img.shape[-1] == self.key_shape_map[key][0]:
                    img = torch.moveaxis(img, -1, 1)
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature.reshape(batch_size, -1))

        for key in self.low_dim_keys:
            data = obs_dict[key]
            batch_size = data.shape[0] if batch_size is None else batch_size
            assert batch_size == data.shape[0]
            if data.ndim == 3:
                assert data.shape[2:] == self.key_shape_map[key]
                data = data.reshape(batch_size, -1)
            else:
                assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {}
        for key, attr in self.shape_meta["obs"].items():
            shape = tuple(attr["shape"])
            horizon = attr.get("horizon", 1)
            example_obs_dict[key] = torch.zeros((1, horizon) + shape, dtype=self.dtype, device=self.device)
        example_output = self.forward(example_obs_dict)
        return example_output.shape[1:]
