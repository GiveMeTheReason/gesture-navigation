import glob
import json
import os

import numpy as np

import open3d as o3d
from model.transforms import PointCloud_To_RGBD

GESTURES_SET = (
    # 'high',
    'start',
    'select',
    # 'swipe_right',
    # 'swipe_left',
)

DATA_DIR = os.path.join(
    os.path.expanduser('~'),
    'personal',
    'gestures_dataset',
    'HuaweiGesturesDataset',
    'undistorted'
)
# DATA_DIR = os.path.join(
#     'D:\\',
#     'GesturesNavigation',
#     'dataset',
# )

RENDER_OPTION = 'render_option.json'
# RENDER_OPTION = os.path.join(
#     'D:\\',
#     'GesturesNavigation',
#     'gestures_navigation',
#     'pc_data',
#     'render_option.json'
# )

CAMERAS_DIR = ('cam_center', 'cam_right', 'cam_left')

CALIBRATION_DIR = os.path.join('pc_data', 'calib_params')
CALIBRATION_INTRINSIC = {
    'cam_center': '1m.json',
    'cam_right': '2s.json',
    'cam_left': '9s.json',
}


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


def main():
    intrinsics_paths = [os.path.join(CALIBRATION_DIR, CALIBRATION_INTRINSIC[camera])
                        for camera in CAMERAS_DIR]
    intrinsics = get_intrinsics(intrinsics_paths)

    *image_size, = map(int, intrinsics[0][:2])

    batch_size = 1

    counter = 0
    for gesture in GESTURES_SET:
        counter += len(glob.glob(os.path.join(DATA_DIR, 'G*', gesture, '*', 'trial*')))
    print(f'Total count of trials: {counter}')
    print(f'Estimated memory usage: {round(counter * 120 / 1024, 2)} GB')
    print(f'Estimated processing time: {round(counter * 120 / 7200, 2)} hours')

    render_option_path = RENDER_OPTION

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=image_size[0], height=image_size[1])

    visualizer.get_render_option().load_from_json(render_option_path)

    angle = np.deg2rad(-30)
    z_target = 1.25

    loc = np.array([0., 0., 0., 0., 0., 0.])
    scale = np.array([np.pi/24, np.pi/18, np.pi/48, 0.2, 0.1, 0.1]) / 1.5

    trans = PointCloud_To_RGBD(
        batch_size,
        intrinsics[0],
        visualizer,
        render_option_path,
        angle=angle,
        z_target=z_target,
        loc=loc,
        scale=scale,
        image_size=None,
    )

    for participant in glob.glob(os.path.join(DATA_DIR, 'G*')):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, '*')):
                for trial in glob.glob(os.path.join(hand, '*')):
                    pc_paths = sorted(glob.glob(os.path.join(trial, '*.pcd')))

                    if not pc_paths:
                        continue

                    save_dir = os.path.join(
                        os.path.expanduser('~'),
                        'personal',
                        'gestures_navigation',
                        'pc_data',
                        'dataset',
                        os.path.split(participant)[-1],
                        gesture,
                        os.path.split(hand)[-1],
                        os.path.split(trial)[-1]
                    )
                    # save_dir = os.path.join(
                    #     'D:\\',
                    #     'GesturesNavigation',
                    #     'dataset',
                    #     os.path.split(participant)[-1],
                    #     gesture,
                    #     os.path.split(hand)[-1],
                    #     os.path.split(trial)[-1]
                    # )

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    for pc_path in pc_paths:
                        rendered_image = trans(pc_path, 0)

                        rgb_image = rendered_image[:3, :, :].permute(1, 2, 0).contiguous()
                        depth_image = rendered_image[3:, :, :].permute(1, 2, 0).contiguous()

                        o3d.io.write_image(
                            os.path.splitext(pc_path)[0] + '.jpg',
                            o3d.geometry.Image((rgb_image * 255).numpy().astype(np.uint8))
                        )
                        o3d.io.write_image(
                            os.path.splitext(pc_path)[0] + '.png',
                            o3d.geometry.Image((depth_image * 255).numpy().astype(np.uint8))
                        )


if __name__ == '__main__':
    main()
