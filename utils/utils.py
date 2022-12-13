import glob
import json
import os
import re
import typing as tp

import mrob
import numpy as np


def estimate_execution_resources(
    dataset_dir: str,
    gestures_set: set
) -> None:
    """
    Prints estimated time and memory for processing the dataset

    Parameters:
    dataset_dir (str): directory of dataset
    gestures_set (set): set of chosen gestures
    """
    counter = 0
    for gesture in gestures_set:
        counter += len(glob.glob(os.path.join(dataset_dir, 'G*', gesture, '*', 'trial*')))
    print(f'Total count of trials: {counter}')
    print(f'Estimated memory usage: {round(counter * 120 / 1024, 2)} GB')
    print(f'Estimated processing time: {round(counter * 120 / 7200, 2)} hours')


def build_intrinsic_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Returns an intrinsic matrix from list of parameters

    Parameters:
    fx (float): focal x in pixels
    fy (float): focal y in pixels
    cx (float): principal point x in pixels
    cy (float): principal point y in pixels

    Returns:
    intrinsic_matrix (np.ndarray[float]): intrinsic matrix
    """
    intrinsic_matrix = np.array([
        [fx, 0., cx, 0.],
        [0., fy, cy, 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ], dtype=np.float64)

    return intrinsic_matrix


def build_extrinsic_matrix(
    angle: float,
    z_target: float,
) -> np.ndarray:
    """
    Returns an extrinsic matrix for new camera position from list of parameters
    Camera position calculates as rotation around X-axis shifted by Z on z_target
    So, camera rotates around z_target point

    Parameters:
    angle (float): angle of rotation in radians
    z_target (float): z-coornate of point of rotation

    Returns:
    extrinsic_matrix (np.ndarray[float]): extrinsic matrix
    """
    extrinsic_matrix = np.array([
        [1., 0., 0., 0.],
        [0., np.cos(angle), -np.sin(angle), z_target * np.sin(angle)],
        [0., np.sin(angle), np.cos(angle), z_target * (1 - np.cos(angle))],
        [0., 0., 0., 1.],
    ], dtype=np.float64)

    return extrinsic_matrix


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


def get_extrinsics(filenames):
    """
    Returns a list [N x 4 x 4] of transforms
    N = number of filenames

    Parameters:
    filenames (Sequence[str]): JSON filenames containing calibration

    Returns:
    extrinsics (np.ndarray[float]): relative calibration parameters
    """
    assert all(filenames)

    extrinsics = np.zeros((len(filenames), 4, 4), dtype=np.float64)

    for i, filename in enumerate(filenames):
        with open(filename) as json_file:
            data = json.load(json_file)

        T = np.eye(4)
        T[:3, :3] = np.array(data['CalibrationInformation']['Cameras'][1]['Rt']['Rotation']).reshape(3, 3)
        T[:3, 3] = np.array(data['CalibrationInformation']['Cameras'][1]['Rt']['Translation'])

        extrinsics[i] = T

    return extrinsics


def randomize_extrinsic(
    extrinsic_matrix: np.ndarray,
    loc: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Returns matrix with applied random normal (mean=0, var=var)
    variations on state vector [th1, th2, th3, x, y, z] using mrob

    Parameters:
    extrinsic_matrix (np.ndarray[float]): extrinsic parameters of camera [4 x 4]
    loc (float or Sequence[float]): mean, can be array of 6 floats
    scale (float or Sequence[float]): standard deviation, can be array of 6 floats

    Returns:
    randomized_extrinsic_matrix (np.ndarray[float]): randomized extrinsic matrix
    """
    T = mrob.geometry.SE3(extrinsic_matrix)

    randomized_vector = np.random.normal(loc=loc, scale=scale)
    randomized_extrinsic_matrix = (mrob.geometry.SE3(randomized_vector) * T).T()

    return randomized_extrinsic_matrix


def find_timestamp(s):
    """
    Returns indexes of start and the end (negative) of the longest numerical sequence
    
    Parameters:
    s (str): string with numerical timestamp

    Returns:
    timestamp_index (list[int]): start and end (negative) of the timestamp sequence
    """
    longest = max(re.findall(r'\d+', s), key=len)
    start = s.find(longest)
    
    timestamp_index = [start, start + len(longest) - len(s)]

    return timestamp_index


def map_nearest(cameras_filenames, main_camera_index=0):
    """
    Returns an array [L x N] of indexes that matches every i frame (along L)
    from main camera to the nearest frame from other cameras by timestamps
    All filenames should contain timestamps and be sorted by them
    N = number of cameras
    L = number of filenames in cameras_filenames[main_camera_index]

    Parameters:
    cameras_filenames (Sequence[Sequence[str]]): filenames from N cameras
    main_camera_index (int, default=0): main camera index

    Returns:
    nearest (np.ndarray[int]): indexes of filenames that are nearest to i frame from main camera
    """
    assert all(cameras_filenames)

    nearest = np.zeros((len(cameras_filenames[main_camera_index]), len(cameras_filenames)), dtype=np.int64)
    pointers = np.zeros(len(cameras_filenames), dtype=np.int64)

    patterns = np.array([find_timestamp(filenames[0]) for filenames in cameras_filenames], dtype=np.int64)

    for i, main_filename in enumerate(cameras_filenames[main_camera_index]):
        now = int(main_filename[patterns[main_camera_index, 0]:patterns[main_camera_index, 1]])

        for j, (pointer, filenames) in enumerate(zip(pointers, cameras_filenames)):
            now_camera = int(filenames[pointer][patterns[j, 0]:patterns[j, 1]])

            while pointer + 1 < len(filenames) and \
                    now - now_camera > int(filenames[pointer + 1][patterns[j, 0]:patterns[j, 1]]) - now:
                pointer += 1
                now_camera = int(filenames[pointer][patterns[j, 0]:patterns[j, 1]])

            nearest[i, j] = pointer

    return nearest







