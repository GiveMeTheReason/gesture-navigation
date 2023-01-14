import glob
import itertools
import os

import numpy as np

import utils.utils as utils
import utils.utils_o3d as utils_o3d

GESTURES_SET = (
    'start',
    'select',
)

# PC_DATA_DIR = os.path.join(
#     os.path.expanduser('~'),
#     'personal',
#     'gestures_dataset',
#     'HuaweiGesturesDataset',
#     'undistorted'
# )
# PC_DATA_DIR = os.path.join(
#     'D:\\',
#     'GesturesNavigation',
#     'dataset',
# )
PC_DATA_DIR = os.path.join(
    '/root',
    'project',
    'gestures_dataset_initial',
)

# SAVE_DIR = os.path.join(
#     os.path.expanduser('~'),
#     'personal',
#     'gestures_navigation',
#     'pc_data',
#     'dataset',
# )
# SAVE_DIR = os.path.join(
#     'D:\\',
#     'GesturesNavigation',
#     'dataset',
# )
SAVE_DIR = os.path.join(
    '/root',
    'project',
    'gestures_dataset_processed',
)

# RENDER_OPTION = 'render_option.json'
RENDER_OPTION = os.path.join(
    os.path.dirname(SAVE_DIR),
    'gestures_navigation',
    'data',
    'render_option.json'
)

CAMERAS_DIR = ('cam_center', 'cam_right', 'cam_left')

CALIBRATION_DIR = os.path.join(
    os.path.dirname(SAVE_DIR),
    'gestures_navigation',
    'data',
    'calib_params',
)
CALIBRATION_INTRINSIC = {
    'cam_center': '1m.json',
    'cam_right': '2s.json',
    'cam_left': '9s.json',
}
CALIBRATION_EXTRINSIC = {
    ('cam_center', 'cam_right'): os.path.join('1-2', 'calibration_blob.json'),
    ('cam_center', 'cam_left'): os.path.join('1-9', 'calibration_blob.json'),
    ('cam_right', 'cam_left'): os.path.join('2-9', 'calibration_blob.json'),
}

main_camera_index = 0


def main():
    intrinsics_paths = [os.path.join(CALIBRATION_DIR, CALIBRATION_INTRINSIC[camera])
                        for camera in CAMERAS_DIR]
    intrinsics = utils.get_intrinsics(intrinsics_paths)

    extrinsics_paths = [os.path.join(CALIBRATION_DIR, CALIBRATION_EXTRINSIC[camera])
                        for camera in itertools.combinations(CAMERAS_DIR, 2)]
    extrinsics = utils.get_extrinsics(extrinsics_paths)

    utils.estimate_execution_resources(PC_DATA_DIR, GESTURES_SET)

    for participant in glob.glob(os.path.join(PC_DATA_DIR, 'G*')):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, '*')):
                for trial in glob.glob(os.path.join(hand, '*')):
                    images_paths = [sorted(glob.glob(os.path.join(trial, camera, type, '*')))
                                    for camera, type in itertools.product(CAMERAS_DIR, ('color', 'depth'))]

                    if not images_paths:
                        continue

                    mapped_indexes = utils.map_nearest(
                        images_paths, main_camera_index)

                    save_dir = os.path.join(
                        SAVE_DIR,
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

                        rgbd_images = utils_o3d.get_rgbd_images(
                            paths,
                            depth_scale=1000,
                            depth_trunc=2.0
                        )

                        point_clouds = utils_o3d.create_point_clouds(
                            rgbd_images,
                            intrinsics[::2],
                            np.concatenate(
                                [[np.eye(4)], extrinsics[:len(CAMERAS_DIR) - 1]])
                        )

                        pc_concatenated = utils_o3d.concatenate_point_clouds(
                            point_clouds)

                        # Filter outliers
                        pc_concatenated, _ = pc_concatenated.remove_radius_outlier(
                            10, 0.01)

                        utils_o3d.write_point_cloud(
                            os.path.join(
                                save_dir,
                                f'{str(group[0]).zfill(5)}.pcd'
                            ),
                            pc_concatenated,
                            compressed=True,
                        )
                    break


if __name__ == '__main__':
    main()
