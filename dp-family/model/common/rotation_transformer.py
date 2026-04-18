from typing import Union
import functools

import numpy as np
import torch
from scipy.spatial.transform import Rotation

try:
    import pytorch3d.transforms as pt
except ImportError:
    pt = None


def _rotation_6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(min=1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def _matrix_to_rotation_6d_np(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x[..., :, 0], x[..., :, 1]), axis=-1)


def _axis_angle_to_matrix_np(x: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(x.reshape(-1, 3)).as_matrix().reshape(x.shape[:-1] + (3, 3))


def _matrix_to_axis_angle_np(x: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(x.reshape(-1, 3, 3)).as_rotvec().reshape(x.shape[:-2] + (3,))


def _quaternion_to_matrix_np(x: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(x.reshape(-1, 4)).as_matrix().reshape(x.shape[:-1] + (3, 3))


def _matrix_to_quaternion_np(x: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(x.reshape(-1, 3, 3)).as_quat().reshape(x.shape[:-2] + (4,))


def _euler_angles_to_matrix_np(x: np.ndarray, convention: str) -> np.ndarray:
    return Rotation.from_euler(convention.lower(), x.reshape(-1, 3)).as_matrix().reshape(x.shape[:-1] + (3, 3))


def _matrix_to_euler_angles_np(x: np.ndarray, convention: str) -> np.ndarray:
    return Rotation.from_matrix(x.reshape(-1, 3, 3)).as_euler(convention.lower()).reshape(x.shape[:-2] + (3,))


class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
            from_rep='axis_angle',
            to_rep='rotation_6d',
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if pt is None:
            if from_rep != 'matrix':
                forward_funcs.append(self._get_np_to_matrix(from_rep, from_convention))
                inverse_funcs.append(self._get_matrix_to_np(from_rep, from_convention))

            if to_rep != 'matrix':
                forward_funcs.append(self._get_matrix_to_np(to_rep, to_convention))
                inverse_funcs.append(self._get_np_to_matrix(to_rep, to_convention))

            self.forward_funcs = forward_funcs
            self.inverse_funcs = inverse_funcs[::-1]
            return

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention)
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention)
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs[::-1]

    @staticmethod
    def _get_np_to_matrix(rep: str, convention=None):
        if rep == 'axis_angle':
            return _axis_angle_to_matrix_np
        if rep == 'quaternion':
            return _quaternion_to_matrix_np
        if rep == 'rotation_6d':
            return _rotation_6d_to_matrix_np
        if rep == 'euler_angles':
            return functools.partial(_euler_angles_to_matrix_np, convention=convention)
        raise ValueError(f'Unsupported rotation representation: {rep}')

    @staticmethod
    def _get_matrix_to_np(rep: str, convention=None):
        if rep == 'axis_angle':
            return _matrix_to_axis_angle_np
        if rep == 'quaternion':
            return _matrix_to_quaternion_np
        if rep == 'rotation_6d':
            return _matrix_to_rotation_6d_np
        if rep == 'euler_angles':
            return functools.partial(_matrix_to_euler_angles_np, convention=convention)
        raise ValueError(f'Unsupported rotation representation: {rep}')

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        if pt is not None:
            x_ = x
            if isinstance(x, np.ndarray):
                x_ = torch.from_numpy(x)
            x_: torch.Tensor
            for func in funcs:
                x_ = func(x_)
            if isinstance(x, np.ndarray):
                return x_.numpy()
            return x_

        input_is_torch = isinstance(x, torch.Tensor)
        device = x.device if input_is_torch else None
        dtype = x.dtype if input_is_torch else None
        x_np = x.detach().cpu().numpy() if input_is_torch else np.asarray(x)
        for func in funcs:
            x_np = func(x_np)
        if input_is_torch:
            return torch.as_tensor(x_np, device=device, dtype=dtype)
        return x_np

    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi, 2*np.pi, size=(1000, 3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
