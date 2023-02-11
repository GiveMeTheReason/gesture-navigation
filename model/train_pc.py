import glob
import os
import random

import numpy as np
import torch
import torch.utils.data

import model.loader as loader
import model.losses as losses
import model.model_cnn as model_cnn
import model.train_loop as train_loop
import model.transforms as transforms
import utils.utils as utils
import utils.utils_o3d as utils_o3d
from config import CONFIG


def main():
    exp_id = CONFIG['experiment_id']
    log_filename = CONFIG['directories']['outputs']['logger_path']
    checkpoint_path = CONFIG['directories']['outputs']['checkpoint_path']

    seed = CONFIG['train']['seed']
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    GESTURES_MAP = CONFIG['gestures']['gestures_set']
    GESTURES_SET = [gesture[0] for gesture in GESTURES_MAP]
    with_rejection = CONFIG['gestures']['with_rejection']

    PC_DATA_DIR = CONFIG['directories']['datasets']['processed_dir']

    cameras = sorted([cam for cam in CONFIG['directories']
                      ['cameras']['cameras'].keys()])[-1:]
    CAMERAS_DIR = [CONFIG['directories']
                   ['cameras']['cameras'][cam]['dir'] for cam in cameras]
    CALIBRATION_INTRINSIC = {CONFIG['directories']
                             ['cameras']['cameras'][cam]['dir']: CONFIG['directories']
                             ['cameras']['cameras'][cam]['intrinsic'] for cam in cameras}

    RENDER_OPTION = CONFIG['directories']['cameras']['render_option']

    main_camera_index = 0

    label_map = {**GESTURES_MAP}
    if CONFIG['gestures']['with_rejection']:
        label_map['no_gesture'] = len(label_map)

    batch_size = CONFIG['train']['batch_size']
    max_workers = CONFIG['train']['max_workers']

    resized_image_size = CONFIG['train']['resized_image_size']
    frames = CONFIG['train']['frames_buffer']
    base_fps = CONFIG['train']['base_fps']
    target_fps = CONFIG['train']['target_fps']

    angle = np.deg2rad(CONFIG['augmentations']['angle'])
    z_target = CONFIG['augmentations']['z_target']

    loc = np.array(CONFIG['augmentations']['loc_angles'] +
                   CONFIG['augmentations']['los_position'])
    scale = np.array(CONFIG['augmentations']['std_angles'] +
                     CONFIG['augmentations']['std_position'])

    lr = CONFIG['train']['lr']
    weight_decay = CONFIG['train']['weight_decay']
    weight = torch.tensor(CONFIG['train']['weights'])

    epochs = CONFIG['train']['epochs']
    validate_each_epoch = CONFIG['train']['validation_epoch_interval']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    intrinsics_paths = [CALIBRATION_INTRINSIC[camera]
                        for camera in CAMERAS_DIR]
    intrinsics = utils.get_intrinsics(intrinsics_paths)[main_camera_index]

    *image_size, = map(int, intrinsics[:2])

    visualizer = utils_o3d.get_visualizer(image_size, RENDER_OPTION)

    pc_to_rgb = transforms.PointCloudToRGBD(
        batch_size,
        intrinsics,
        visualizer,
        angle=angle,
        z_target=z_target,
        loc=loc,
        scale=scale,
        rgb_transforms=transforms.TrainRGBTransforms(resized_image_size),
        depth_transforms=transforms.TrainDepthTransforms(
            resized_image_size, with_inverse=True),
    )
    rgb_depth_to_rgb = transforms.RGBDepthToRGBD(
        rgb_transforms=transforms.TestRGBTransforms(resized_image_size),
        depth_transforms=transforms.TestDepthTransforms(
            resized_image_size, with_inverse=True),
    )

    data_list = [
        d for d in glob.glob(os.path.join(PC_DATA_DIR, 'G*/*/*/*'))
        if d.split(os.path.sep)[-3] in GESTURES_SET
    ]
    train_len = int(
        CONFIG['train']['train_ratio'] * len(data_list))
    test_len = len(data_list) - train_len
    train_list, test_list = map(
        list, torch.utils.data.random_split(data_list, [train_len, test_len]))

    # train_list = train_list[:1]
    # test_list = test_list[:1]

    train_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=train_list,
        label_map=label_map,
        transforms=pc_to_rgb,
        base_fps=base_fps,
        target_fps=target_fps,
        data_type=loader.AllowedDatasets.PCD,
        with_rejection=with_rejection,
    )
    train_loader = loader.MultiStreamDataLoader(
        train_datasets, image_size=resized_image_size, num_workers=0)

    test_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=test_list,
        label_map=label_map,
        transforms=rgb_depth_to_rgb,
        base_fps=base_fps,
        target_fps=target_fps,
        data_type=loader.AllowedDatasets.PROXY,
        with_rejection=with_rejection,
    )
    test_loader = loader.MultiStreamDataLoader(
        test_datasets, image_size=resized_image_size, num_workers=1)

    model = model_cnn.CNNClassifier(
        resized_image_size,
        frames=frames,
        batch_size=batch_size,
        num_classes=len(label_map),
    )
    model.to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = losses.CrossEntropyLoss(weight=weight)
    loss_func.to(device)

    train_loop.train_model(
        model,
        train_loader,
        test_loader,
        train_list,
        test_list,
        label_map,
        optimizer,
        loss_func,
        epochs,
        validate_each_epoch,
        target_fps,
        base_fps,
        checkpoint_path,
        log_filename,
        device,
    )

    visualizer.destroy_window()


if __name__ == '__main__':
    main()
