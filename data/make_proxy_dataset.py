import glob
import os

import numpy as np

import model.transforms as transforms
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
PC_DATA_DIR = os.path.join(
    'D:\\',
    'GesturesNavigation',
    'dataset',
)

# SAVE_DIR = os.path.join(
#     os.path.expanduser('~'),
#     'personal',
#     'gestures_navigation',
#     'pc_data',
#     'dataset',
# )
SAVE_DIR = os.path.join(
    'D:\\',
    'GesturesNavigation',
    'dataset',
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


def main():
    intrinsics_paths = [os.path.join(CALIBRATION_DIR, CALIBRATION_INTRINSIC[camera])
                        for camera in CAMERAS_DIR]
    intrinsics = utils.get_intrinsics(intrinsics_paths)

    *image_size, = map(int, intrinsics[0][:2])

    batch_size = 1

    utils.estimate_execution_resources(PC_DATA_DIR, GESTURES_SET)

    visualizer = utils_o3d.get_visualizer(image_size, RENDER_OPTION)

    angle = np.deg2rad(-30)
    z_target = 1.25

    loc = np.array([0., 0., 0., 0., 0., 0.])
    scale = np.array([np.pi/24, np.pi/18, np.pi/48, 0.2, 0.1, 0.1]) / 1.5

    img_transforms = transforms.PointCloudToRGBD(
        batch_size,
        intrinsics[0],
        visualizer,
        RENDER_OPTION,
        angle=angle,
        z_target=z_target,
        loc=loc,
        scale=scale,
        image_size=None,
    )

    for participant in glob.glob(os.path.join(PC_DATA_DIR, 'G*')):
        for gesture in GESTURES_SET:
            for hand in glob.glob(os.path.join(participant, gesture, '*')):
                for trial in glob.glob(os.path.join(hand, '*')):
                    pc_paths = sorted(glob.glob(os.path.join(trial, '*.pcd')))

                    if not pc_paths:
                        continue

                    save_dir = os.path.join(
                        SAVE_DIR,
                        os.path.split(participant)[-1],
                        gesture,
                        os.path.split(hand)[-1],
                        os.path.split(trial)[-1]
                    )

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    for pc_path in pc_paths:
                        rendered_image = img_transforms(pc_path, 0)

                        rgb_image = rendered_image[:3, :, :].permute(1, 2, 0).contiguous()
                        depth_image = rendered_image[3:, :, :].permute(1, 2, 0).contiguous()

                        utils_o3d.write_image(
                            os.path.splitext(pc_path)[0] + '.jpg',
                            (rgb_image * 255).numpy().astype(np.uint8),
                        )
                        utils_o3d.write_image(
                            os.path.splitext(pc_path)[0] + '.png',
                            (depth_image * 255).numpy().astype(np.uint8),
                        )

    visualizer.destroy_window()


if __name__ == '__main__':
    main()
