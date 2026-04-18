import copy
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from termcolor import cprint

from common.pytorch_util import replace_submodules
from model.common.module_attr_mixin import ModuleAttrMixin


logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spatial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        model_name: str,
        pretrained: bool,
        frozen: bool,
        global_pool: str,
        transforms: list,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        feature_aggregation: str = "spatial_embedding",
        downsample_ratio: int = 32,
        position_encording: str = "learnable",
    ):
        super().__init__()

        del imagenet_norm

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        assert global_pool == ""

        if model_name == "r3m":
            if pretrained:
                from r3m import load_r3m

                original_home = os.environ.get("HOME")
                r3m_home = os.environ.get("R3M_HOME")
                try:
                    if r3m_home:
                        os.environ["HOME"] = r3m_home
                    model = load_r3m("resnet18", pretrained=True)
                finally:
                    if r3m_home:
                        if original_home is None:
                            os.environ.pop("HOME", None)
                        else:
                            os.environ["HOME"] = original_home
            else:
                from r3m.models.models_r3m import R3M

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = R3M(
                    device=device,
                    lr=0.0,
                    hidden_dim=512,
                    size=18,
                    l2weight=1.0,
                    l1weight=1.0,
                    langweight=0.0,
                    tcnweight=0.0,
                    l2dist=True,
                    bs=16,
                )
            model.eval()
            cprint(f"Loaded R3M model using {model_name}. pretrained={pretrained}", "green")
            feature_dim = 512
        else:
            raise NotImplementedError(f"Unsupported model_name: {model_name}")

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16)
                    if (x.num_features % 16 == 0)
                    else (x.num_features // 8),
                    num_channels=x.num_features,
                ),
            )

        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            attr_type = attr.get("type", "low_dim")
            if attr_type == "rgb":
                assert image_shape is None or image_shape == shape
                image_shape = shape

        if transforms is not None and len(transforms) > 0 and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == "RandomCrop"
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[1] * ratio)),
                torchvision.transforms.Resize(size=image_shape[1], antialias=True),
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            attr_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if attr_type == "rgb":
                rgb_keys.append(key)
                key_model_map[key] = model if share_rgb_model else copy.deepcopy(model)
                key_transform_map[key] = transform
            elif attr_type == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {attr_type}")

        feature_map_shape = [x // downsample_ratio for x in image_shape[1:]]
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print("rgb keys:         ", rgb_keys)
        print("low_dim_keys keys:", low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation

        if self.feature_aggregation == "soft_attention":
            self.attention = nn.Sequential(nn.Linear(feature_dim, 1, bias=False), nn.Softmax(dim=1))
        elif self.feature_aggregation == "spatial_embedding":
            self.spatial_embedding = torch.nn.Parameter(
                torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim)
            )
        elif self.feature_aggregation == "transformer":
            if position_encording == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim)
                )
            elif position_encording == "sinusoidal":
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim)
                )
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            else:
                raise ValueError(f"Unsupported position_encording: {position_encording}")
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4,
            )
        elif self.feature_aggregation == "attention_pool_2d":
            self.attention_pool_2d = AttentionPool2d(
                spatial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim,
            )

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def aggregate_feature(self, feature):
        if self.model_name == "r3m":
            return feature

        assert len(feature.shape) == 4
        if self.feature_aggregation == "attention_pool_2d":
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2)
        feature = torch.transpose(feature, 1, 2)

        if self.feature_aggregation == "avg":
            return torch.mean(feature, dim=[1])
        if self.feature_aggregation == "max":
            return torch.amax(feature, dim=[1])
        if self.feature_aggregation == "soft_attention":
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        if self.feature_aggregation == "spatial_embedding":
            return torch.mean(feature * self.spatial_embedding, dim=1)
        if self.feature_aggregation == "transformer":
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        assert self.feature_aggregation is None
        return feature

    def forward(self, obs_dict):
        features = []
        batch_size = next(iter(obs_dict.values())).shape[0]

        for key in self.rgb_keys:
            img = obs_dict[key]
            batch, steps = img.shape[:2]
            assert batch == batch_size
            img = img.reshape(batch * steps, *img.shape[2:])

            if img.shape[1:] != self.key_shape_map[key]:
                target_h, target_w = self.key_shape_map[key][1], self.key_shape_map[key][2]
                img = F.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img).to(self.device)
            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == batch * steps
            features.append(feature.reshape(batch, -1))

        for key in self.low_dim_keys:
            data = obs_dict[key]
            batch, steps = data.shape[:2]
            assert batch == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(batch, -1))

        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros((1, attr["horizon"]) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        return example_output.shape
