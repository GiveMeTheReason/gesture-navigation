import glob
import os

import numpy as np
import PIL.PngImagePlugin as PngIP
import torch
import torchvision.transforms as T

import model.transforms as transforms
import utils.utils as utils
import utils.utils_o3d as utils_o3d
from config import CONFIG

GESTURES_MAP = CONFIG['gestures']['gestures_set']
GESTURES_SET = [gesture[0] for gesture in GESTURES_MAP]

PC_DATA_DIR = CONFIG['directories']['datasets']['processed_dir']
SAVE_DIR = CONFIG['directories']['datasets']['processed_dir']

RENDER_OPTION = CONFIG['directories']['cameras']['render_option']

cameras = sorted([cam for cam in CONFIG['directories']
                 ['cameras']['cameras'].keys()])
cameras = cameras[-1:] + cameras[:-1]
CAMERAS_DIR = [CONFIG['directories']
               ['cameras']['cameras'][cam]['dir'] for cam in cameras]

CALIBRATION_INTRINSIC = {CONFIG['directories']
                         ['cameras']['cameras'][cam]['dir']: CONFIG['directories']
                         ['cameras']['cameras'][cam]['intrinsic'] for cam in cameras}


def main():
    intrinsics_paths = [CALIBRATION_INTRINSIC[camera]
                        for camera in CAMERAS_DIR]
    intrinsics = utils.get_intrinsics(intrinsics_paths)

    *image_size, = map(int, intrinsics[0][:2])

    batch_size = 1

    utils.estimate_execution_resources(
        PC_DATA_DIR, GESTURES_SET, is_proxy=True)

    visualizer = utils_o3d.get_visualizer(image_size, RENDER_OPTION)
    rgb_to_pil = T.ToPILImage(mode='RGB')
    depth_to_pil = T.ToPILImage(mode='L')

    angle = np.deg2rad(CONFIG['augmentations']['angle'])
    z_target = CONFIG['augmentations']['z_target']

    loc = np.array(CONFIG['augmentations']['loc_angles'] +
                   CONFIG['augmentations']['los_position'])
    scale = np.array(CONFIG['augmentations']['std_angles'] +
                     CONFIG['augmentations']['std_position'])

    img_transforms = transforms.PointCloudToRGBD(
        batch_size,
        intrinsics[0],
        visualizer,
        angle=angle,
        z_target=z_target,
        loc=loc,
        scale=scale,
        rgb_transforms=None,
        depth_transforms=None,
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
                        base_filename = os.path.join(
                            save_dir, os.path.splitext(os.path.basename(pc_path))[0])
                        if os.path.exists(base_filename + '.jpg') and os.path.exists(base_filename + '.png'):
                            continue

                        rendered_image = img_transforms(pc_path, 0)

                        rgb_image = rendered_image[:3, :, :]
                        depth_image = rendered_image[3:, :, :]

                        depth_max = depth_image.max()
                        depth_image = (depth_image / depth_max *
                                       255).type(torch.uint8)
                        depth_metadata = PngIP.PngInfo()
                        depth_metadata.add_text(
                            'MaxDepth', str(depth_max.item()))

                        rgb_to_pil(rgb_image).save(
                            base_filename + '.jpg',
                        )
                        depth_to_pil(depth_image).save(
                            base_filename + '.png',
                            pnginfo=depth_metadata,
                        )

    visualizer.destroy_window()


if __name__ == '__main__':
    main()
