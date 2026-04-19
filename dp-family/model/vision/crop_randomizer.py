import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf

import model.common.tensor_util as tu


class CropRandomizer(nn.Module):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        super().__init__()

        assert len(input_shape) == 3
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def forward_in(self, inputs):
        assert len(inputs.shape) >= 3
        if self.training:
            out, _ = sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            return tu.join_dimensions(out, 0, 1)

        out = ttf.center_crop(
            img=inputs,
            output_size=(self.crop_height, self.crop_width),
        )
        if self.num_crops > 1:
            batch_size, channels, height, width = out.shape
            out = (
                out.unsqueeze(1)
                .expand(batch_size, self.num_crops, channels, height, width)
                .reshape(-1, channels, height, width)
            )
        return out

    def forward_out(self, inputs):
        if self.num_crops <= 1:
            return inputs
        batch_size = inputs.shape[0] // self.num_crops
        out = tu.reshape_dimensions(
            inputs,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size, self.num_crops),
        )
        return out.mean(dim=1)

    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        header = str(self.__class__.__name__)
        return (
            f"{header}(input_shape={self.input_shape}, "
            f"crop_size=[{self.crop_height}, {self.crop_width}], "
            f"num_crops={self.num_crops})"
        )


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (ndim_im_shape == ndim_indices_shape + 2)

    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_channels, image_height, image_width = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_height - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_width - crop_width)).all().item()

    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = tu.unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = tu.unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    crop_in_grid = torch.cat(
        (crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)),
        dim=-1,
    )

    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(grid_reshape)
    all_crop_inds = all_crop_inds[..., 0] * image_width + all_crop_inds[..., 1]
    all_crop_inds = tu.unsqueeze_expand_at(all_crop_inds, size=image_channels, dim=-3)
    all_crop_inds = tu.flatten(all_crop_inds, begin_axis=-2)

    images_to_crop = tu.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = tu.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    reshape_axis = len(crops.shape) - 1
    crops = tu.reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )

    if is_padded:
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    device = images.device
    source_im = images
    if pos_enc:
        height, width = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(height), torch.arange(width))
        pos_y = pos_y.float().to(device) / float(height)
        pos_x = pos_x.float().to(device) / float(width)
        position_enc = torch.stack((pos_y, pos_x))
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)
        source_im = torch.cat((source_im, position_enc), dim=-3)

    max_sample_h = images.shape[-2] - crop_height
    max_sample_w = images.shape[-1] - crop_width
    leading_shape = images.shape[:-3]
    crop_inds_h = torch.randint(
        low=0,
        high=max_sample_h,
        size=leading_shape + (num_crops,),
        device=device,
    )
    crop_inds_w = torch.randint(
        low=0,
        high=max_sample_w,
        size=leading_shape + (num_crops,),
        device=device,
    )
    crop_inds = torch.stack((crop_inds_h, crop_inds_w), dim=-1)
    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )
    return crops, crop_inds
