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


def main():
    exp_id = '01'
    log_filename = os.path.join('outputs', f'train_log{exp_id}.txt')
    checkpoint_path = os.path.join('outputs', f'checkpoint{exp_id}.pth')

    seed = 0
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    GESTURES_SET = (
        'start',
        'select',
    )
    with_rejection = True

    # PC_DATA_DIR = os.path.join(
    #     os.path.expanduser('~'),
    #     'personal',
    #     'gestures_navigation',
    #     'pc_data',
    #     'dataset',
    # )
    PC_DATA_DIR = os.path.join(
        'D:\\',
        'GesturesNavigation',
        'dataset',
    )

    label_map = {gesture: i for i, gesture in enumerate(GESTURES_SET)}
    if with_rejection:
        label_map['no_gesture'] = len(label_map)

    batch_size = 24
    max_workers = 2

    frames = 1
    base_fps = 30
    target_fps = 5
    resized_image_size = (72, 128)

    lr = 1e-4
    weight_decay = 1e-4
    weight = torch.tensor([1., 1., 1.])

    epochs = 20
    validate_each_epoch = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_rgb_depth_to_rgb = transforms.RGBDepthToRGBD(
        rgb_transforms=transforms.TrainRGBTransforms(resized_image_size),
        depth_transforms=transforms.TrainDepthTransforms(resized_image_size, with_inverse=True),
    )
    test_rgb_depth_to_rgb = transforms.RGBDepthToRGBD(
        rgb_transforms=transforms.TestRGBTransforms(resized_image_size),
        depth_transforms=transforms.TestDepthTransforms(resized_image_size, with_inverse=True),
    )

    data_list = [
        d for d in glob.glob(os.path.join(PC_DATA_DIR, 'G*/*/*/*'))
        if d.split(os.path.sep)[-3] in GESTURES_SET
    ]
    train_len = int(0.75 * len(data_list))
    test_len = len(data_list) - train_len
    train_list, test_list = map(list, torch.utils.data.random_split(data_list, [train_len, test_len]))

    # train_list = train_list[:1]
    # test_list = test_list[:1]

    train_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=train_list,
        label_map=label_map,
        transforms=train_rgb_depth_to_rgb,
        base_fps=base_fps,
        target_fps=target_fps,
        data_type=loader.AllowedDatasets.PROXY,
        with_rejection=with_rejection,
    )
    train_loader = loader.MultiStreamDataLoader(train_datasets, image_size=resized_image_size)

    test_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=test_list,
        label_map=label_map,
        transforms=test_rgb_depth_to_rgb,
        base_fps=base_fps,
        target_fps=target_fps,
        data_type=loader.AllowedDatasets.PROXY,
        with_rejection=with_rejection,
    )
    test_loader = loader.MultiStreamDataLoader(test_datasets, image_size=resized_image_size)

    model = model_cnn.CNNClassifier(
        resized_image_size,
        frames=frames,
        batch_size=batch_size,
        num_classes=len(label_map),
    )
    model.to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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


if __name__ == '__main__':
    main()
