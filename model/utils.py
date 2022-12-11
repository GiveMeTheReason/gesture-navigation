import json
import typing as tp

import numpy as np


def get_intrinsics(filenames: tp.List[str]) -> np.ndarray:
    """
    Returns a list [2*N x 6] of calibration parameters for color and depth cameras
    in form of open3d.camera.PinholeCameraIntrinsic from JSON files
    [width, height, fx, fy, cx, cy]
    N = number of filenames

    Parameters:
    filenames (Sequence[str]): JSON filenames containing calibration

    Returns:
    intrinsics (np.ndarray[float]): calibration parameters
    """
    assert all(filenames)

    intrinsics = np.zeros((2 * len(filenames), 6), dtype=np.float64)

    for i, filename in enumerate(filenames):
        with open(filename) as json_file:
            data = json.load(json_file)

        for j, camera in enumerate(('color', 'depth')):
            intrinsics[2 * i + j] = [
                data[f'{camera}_camera']['resolution_width'],
                data[f'{camera}_camera']['resolution_height'],
                data[f'{camera}_camera']['intrinsics']['parameters']['parameters_as_dict']['fx'],
                data[f'{camera}_camera']['intrinsics']['parameters']['parameters_as_dict']['fy'],
                data[f'{camera}_camera']['intrinsics']['parameters']['parameters_as_dict']['cx'],
                data[f'{camera}_camera']['intrinsics']['parameters']['parameters_as_dict']['cy'],
            ]

    return intrinsics
