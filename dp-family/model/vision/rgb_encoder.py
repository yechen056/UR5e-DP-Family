import torch
import torch.nn as nn
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon
import robomimic.utils.obs_utils as ObsUtils

from common.pytorch_util import replace_submodules
from model.vision.crop_randomizer import CropRandomizer
from model.vision.sensory_encoder import BaseSensoryEncoder


class RobomimicRgbEncoder(BaseSensoryEncoder):
    def __init__(
        self,
        shape_meta: dict,
        crop_shape=None,
        use_group_norm=True,
        eval_fixed_crop=False,
        share_rgb_model=False,
    ):
        super().__init__()

        rgb_ports = []
        port_shape = {}
        for key, attr in shape_meta["obs"].items():
            obs_type = attr["type"]
            shape = attr["shape"]
            if obs_type == "rgb" and "tactile" not in key.lower():
                rgb_ports.append(key)
                port_shape[key] = shape

        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": rgb_ports})

        def crop_randomizer(shape, crop_shape_value):
            if crop_shape_value is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=shape,
                crop_height=crop_shape_value[0],
                crop_width=crop_shape_value[1],
                num_crops=1,
                pos_enc=False,
            )

        def visual_net(shape, crop_shape_value):
            if crop_shape_value is not None:
                shape = (shape[0], crop_shape_value[0], crop_shape_value[1])
            return rmbn.VisualCore(
                input_shape=shape,
                feature_dimension=64,
                backbone_class="ResNet18Conv",
                backbone_kwargs={
                    "input_channels": shape[0],
                    "input_coord_conv": False,
                },
                pool_class="SpatialSoftmax",
                pool_kwargs={
                    "num_kp": 32,
                    "temperature": 1.0,
                    "noise": 0.0,
                },
                flatten=True,
            )

        obs_encoder = rmon.ObservationEncoder()
        if share_rgb_model:
            shared_shape = port_shape[rgb_ports[0]]
            net = visual_net(shared_shape, crop_shape)
            obs_encoder.register_obs_key(
                name=rgb_ports[0],
                shape=shared_shape,
                net=net,
                randomizer=crop_randomizer(shared_shape, crop_shape),
            )
            for port in rgb_ports[1:]:
                assert port_shape[port] == shared_shape
                obs_encoder.register_obs_key(
                    name=port,
                    shape=shared_shape,
                    randomizer=crop_randomizer(shared_shape, crop_shape),
                    share_net_from=rgb_ports[0],
                )
        else:
            for port in rgb_ports:
                shape = port_shape[port]
                net = visual_net(shape, crop_shape)
                obs_encoder.register_obs_key(
                    name=port,
                    shape=shape,
                    net=net,
                    randomizer=crop_randomizer(shape, crop_shape),
                )

        if use_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=module.num_features // 16,
                    num_channels=module.num_features,
                ),
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda module: isinstance(module, rmbn.CropRandomizer),
                func=lambda module: CropRandomizer(
                    input_shape=module.input_shape,
                    crop_height=module.crop_height,
                    crop_width=module.crop_width,
                    num_crops=module.num_crops,
                    pos_enc=module.pos_enc,
                ),
            )

        obs_encoder.make()
        self.encoder = obs_encoder
        self.rgb_keys = list(obs_encoder.obs_shapes.keys())

    def forward(self, obs_dict):
        output = self.encoder(obs_dict)
        batch_size = output.shape[0]
        num_keys = len(self.rgb_keys)
        output = output.reshape(batch_size, num_keys, -1)
        return dict(zip(self.rgb_keys, output.unbind(1)))

    @torch.no_grad()
    def output_feature_dim(self):
        feature_dim = self.encoder.output_shape()[0]
        num_keys = len(self.rgb_keys)
        return dict(zip(self.rgb_keys, [feature_dim // num_keys] * num_keys))

    def modalities(self):
        return ["rgb"]
