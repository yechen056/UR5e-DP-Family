import numpy as np

def get_xyz_from_depth(depth_map, camera_intrinsics, R=None, t=None, camera_height=84, camera_width=84):
    """
    Converts uv-map (obtained using camera height and width) and depth-map to (x,y,z) coordinates
    whose units are same as in the depth map.
    Args:
        camera_name (str)
        camera_height (int)
        camera_width (int)
        depth_map (2x2 array): depth map (normalized returned by mujoco)
    """
    def parse_intrinsics(K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        return fx, fy, cx, cy
    fx, fy, cx, cy = parse_intrinsics(camera_intrinsics)
    uv = np.mgrid[0:camera_height, 0:camera_width]
    x = (uv[1] - cx) / fx * depth_map
    y = (uv[0] - cy) / fy * depth_map
    xyz = np.stack((x, y, depth_map), axis=-1)

    xyz = xyz.reshape(-1, 3)
    if R is not None and t is not None:
        xyz = np.matmul(xyz, R.T) + t
    return xyz
    
    