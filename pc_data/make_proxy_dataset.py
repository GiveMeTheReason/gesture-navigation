import glob
import itertools
import json
import os
import re

import numpy as np

import open3d as o3d

GESTURES_SET = (
    # "high",
    "start",
    "select",
    # "swipe_right",
    # "swipe_left",
)

DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "personal",
    "gestures_dataset",
    "HuaweiGesturesDataset",
    "undistorted"
)

CAMERAS_DIR = ("cam_center", "cam_right", "cam_left")

CALIBRATION_DIR = os.path.join("pc_data", "calib_params")
CALIBRATION_INTRINSIC = {
    "cam_center": "1m.json",
    "cam_right": "2s.json",
    "cam_left": "9s.json",
}
CALIBRATION_EXTRINSIC = {
    ("cam_center", "cam_right"): os.path.join("1-2", "calibration_blob.json"),
    ("cam_center", "cam_left"): os.path.join("1-9", "calibration_blob.json"),
    ("cam_right", "cam_left"): os.path.join("2-9", "calibration_blob.json"),
}

main_camera_index = 0


def get_intrinsics(filenames):
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

        for j, camera in enumerate(("color", "depth")):
            intrinsics[2 * i + j] = [
                data[f"{camera}_camera"]["resolution_width"],
                data[f"{camera}_camera"]["resolution_height"],
                data[f"{camera}_camera"]["intrinsics"]["parameters"]["parameters_as_dict"]["fx"],
                data[f"{camera}_camera"]["intrinsics"]["parameters"]["parameters_as_dict"]["fy"],
                data[f"{camera}_camera"]["intrinsics"]["parameters"]["parameters_as_dict"]["cx"],
                data[f"{camera}_camera"]["intrinsics"]["parameters"]["parameters_as_dict"]["cy"],
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


def get_rgbd_images(images_paths, depth_scale=1000, depth_trunc=5.0):
    """
    Returns a list [N // 2] of open3d.geometry.RGBDImage
    Input list should contain paths in the same order as all pipeline
    N = number of paths

    Parameters:
    images_paths (Sequence[str]): filenames for color and depth images
    depth_scale (float, default=1000.0): ratio to scale depth
    depth_trunc (float, default=5.0): values larger depth_trunc gets truncated.
        The depth values will first be scaled and then truncated.

    Returns:
    rgbd_images (list[open3d.geometry.RGBDImage]): RGB-D images
    """
    assert all(images_paths)
    assert len(images_paths) % 2 == 0
    
    rgbd_images = [[] for _ in range(len(images_paths) // 2)]

    for i in range(len(rgbd_images)):
        rgbd_images[i] = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.io.read_image(images_paths[2 * i]),
            depth=o3d.io.read_image(images_paths[2 * i + 1]),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
    
    return rgbd_images


def create_point_clouds(rgbd_images, intrinsics, extrinsics):
    """
    Returns a list [N] of open3d.geometry.PointCloud
    Input list should contain RGB-D images as open3d.geometry.RGBDImage
    All inputs should have the same shape
    N = number of images

    Parameters:
    rgbd_images (Sequence[open3d.geometry.RGBDImage]): RGB-D images
    intrinsics (np.ndarray[float]): intrinsics of RGB-D images
    extrinsics (np.ndarray[float]): extrinsics of RGB-D images

    Returns:
    rgbd_images (list[open3d.geometry.RGBDImage]): RGB-D images
    """
    assert len(rgbd_images) == len(intrinsics) == len(extrinsics)
    assert all(rgbd_images)

    point_clouds = [[] for _ in range(len(rgbd_images))]
    to_filter = [1, 1, 0, 0, 0, 0]

    for i in range(len(point_clouds)):
        point_clouds[i] = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_images[i],
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                *map(lambda val: int(val[1]) if to_filter[val[0]] else val[1], enumerate([*intrinsics[i]]))
            ),
            extrinsic=np.linalg.inv(extrinsics[i]),
            project_valid_depth_only=True
        )
    
    return point_clouds


def concatenate_point_clouds(point_clouds):
    """
    Returns concateneted point cloud from list of point clouds

    Parameters:
    point_clouds (Sequence[open3d.geometry.PointCloud]): list of point clouds

    Returns:
    concatenated_point_cloud (open3d.geometry.PointCloud): concatenated point cloud
    """
    concatenated_point_cloud = np.sum(point_clouds)
    
    return concatenated_point_cloud


def main():
    intrinsics_paths = [os.path.join(CALIBRATION_DIR, CALIBRATION_INTRINSIC[camera])
                        for camera in CAMERAS_DIR]
    intrinsics = get_intrinsics(intrinsics_paths)

    extrinsics_paths = [os.path.join(CALIBRATION_DIR, CALIBRATION_EXTRINSIC[camera])
                        for camera in itertools.combinations(CAMERAS_DIR, 2)]
    extrinsics = get_extrinsics(extrinsics_paths)


    counter = 0
    for gesture in GESTURES_SET:
        counter += len(glob.glob(os.path.join(DATA_DIR, "G*", gesture, "*", "trial1")))
    print(f"Total count of trials: {counter}")
    print(f"Estimated memory usage: {round(counter * 120 / 1024, 2)} GB")
    print(f"Estimated processing time: {round(counter * 120 / 7200, 2)} hours")


    for participant in glob.glob(os.path.join(DATA_DIR, "G*")):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, "*")):
                for trial in glob.glob(os.path.join(hand, "*")):
                    images_paths = [sorted(glob.glob(os.path.join(trial, camera, type, '*')))
                                    for camera, type in itertools.product(CAMERAS_DIR, ('color', 'depth'))]

                    if not images_paths:
                        continue

                    mapped_indexes = map_nearest(images_paths, main_camera_index)

                    save_dir = os.path.join(
                        os.path.expanduser("~"),
                        "personal",
                        "gestures_navigation",
                        "pc_data",
                        "dataset",
                        os.path.split(participant)[-1],
                        gesture,
                        os.path.split(hand)[-1],
                        os.path.split(trial)[-1]
                    )

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    for group in mapped_indexes:
                        paths = [''] * len(group)
                        for i, index in enumerate(group):
                            paths[i] = images_paths[i][index]

                        rgbd_images = get_rgbd_images(
                            paths,
                            depth_scale=1000,
                            depth_trunc=2.0
                        )

                        point_clouds = create_point_clouds(
                            rgbd_images,
                            intrinsics[::2],
                            np.concatenate([[np.eye(4)], extrinsics[:len(CAMERAS_DIR) - 1]])
                        )

                        pc_concatenated = concatenate_point_clouds(point_clouds)

                        # Filter outliers
                        pc_concatenated, _ = pc_concatenated.remove_radius_outlier(10, 0.01)

                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_dir,
                                f"{str(group[0]).zfill(5)}.pcd"
                            ),
                            pc_concatenated,
                            compressed=True
                        )
                    break


if __name__ == "__main__":
    main()
