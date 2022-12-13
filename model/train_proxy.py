import glob
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.utils.data

import model.loader as loader
import model.losses as losses
import model.model_cnn as model_cnn
import model.transforms as transforms


def main():
    exp_id = '02'
    log_filename = f'train_log{exp_id}.txt'
    checkpoint_path = f'checkpoint{exp_id}.pth'

    seed = 0
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    GESTURES_SET = (
        # 'high',
        'start',
        'select',
        # 'swipe_right',
        # 'swipe_left',
    )
    with_rejection = True

    # PC_DATA_DIR = os.path.join(
    #     os.path.expanduser('~'),
    #     'personal',
    #     'gestures_navigation',
    #     'pc_data',
    #     'dataset'
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

    label_map = {gesture: i for i, gesture in enumerate(GESTURES_SET)}

    batch_size = 2
    max_workers = 2

    frames = 1
    base_fps = 30
    target_fps = 30
    resized_image_size = (72, 128)

    lr = 1e-4
    weight_decay = 1e-3
    weight = torch.tensor([1., 1., 1.])

    epochs = 1
    validate_each_epoch = 1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_list = [
        d for d in glob.glob(os.path.join(PC_DATA_DIR, 'G*/*/*/*'))
        if d.split(os.path.sep)[-3] in GESTURES_SET
    ]
    train_len = int(0.75 * len(data_list))
    test_len = len(data_list) - train_len
    train_list, test_list = map(list, torch.utils.data.random_split(data_list, [train_len, test_len]))

    # train_list = train_list[:]
    # test_list = test_list[:]

    rgb_depth_to_rgb = transforms.RGBDepthToRGBD(
        resized_image_size,
    )

    train_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=train_list,
        label_map=label_map,
        transforms=rgb_depth_to_rgb,
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
        transforms=rgb_depth_to_rgb,
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

    time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    msg = f'{time} | Training for {epochs} epochs started!\n\
        Train set length: {len(train_list)}\n\
        Test set length: {len(test_list)}\n'.replace('  ', '')
    with open(log_filename, 'a', encoding='utf-8') as log_file:
        log_file.write(msg)
    print(msg)

    for epoch in range(epochs):

        model.train()
        train_accuracy = 0
        train_loss = 0
        n = 0

        confusion_matrix_train = torch.zeros((len(label_map), len(label_map)), dtype=torch.int)

        counter = 0
        for images, labels in train_loader:
            # print(labels)
            # print(pc_paths)
            # if not all(labels):
            #     continue

            # for i, img in enumerate(pc_paths):
            #     plt.imsave(f'{i}_color.png', img[:3].permute(1, 2, 0).numpy())
            #     plt.imsave(f'{i}_depth.png', img[-1].numpy())
            counter += 1

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            prediction = model(images)
            batch_loss = loss_func(prediction, labels)
            batch_loss.backward()
            optimizer.step()

            print(f'{datetime.now().strftime("%Y.%m.%d %H:%M:%S")} TRAIN\n{epoch=}, {counter=}/{len(train_list)*120*target_fps//base_fps}\n{prediction=}\n{labels=}')

            prediction_probs, prediction_labels = prediction.max(1)
            train_accuracy += (prediction_labels == labels).sum().float()
            train_loss += batch_loss.item()
            n += len(labels)

            pred = torch.argmax(prediction, dim=1)
            for i in range(len(labels)):
                confusion_matrix_train[pred[i], labels[i]] += 1

            # break

        train_accuracy /= n
        train_loss /= len(train_list)

        time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        msg = f'{time} [Epoch: {epoch+1:02}] Train acc: {train_accuracy:.4f} | loss: {train_loss:.4f}\n\
            {confusion_matrix_train=}\n'.replace('  ', '')
        with open(log_filename, 'a', encoding='utf-8') as log_file:
            log_file.write(msg)
        print(msg)

        torch.save(model.state_dict(), checkpoint_path)

        if (epoch+1) % validate_each_epoch == 0:
            model.eval()
            val_accuracy = 0
            val_loss = 0
            n = 0

            confusion_matrix_val = torch.zeros((len(label_map), len(label_map)), dtype=torch.int)

            counter = 0
            with torch.no_grad():
                for val_images, val_labels in test_loader:
                    counter += 1

                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    prediction = model(val_images)
                    prediction_probs, prediction_labels = prediction.max(1)

                    print(f'{datetime.now().strftime("%Y.%m.%d %H:%M:%S")} VAL\n{epoch=}, {counter=}/{len(test_list)*120*target_fps//base_fps}\n{prediction=}\n{val_labels=}')

                    val_accuracy += (prediction_labels == val_labels).sum().float()
                    val_loss += loss_func(prediction, val_labels).item()
                    n += len(val_labels)

                    pred = torch.argmax(prediction, dim=1)
                    for i in range(len(val_labels)):
                        confusion_matrix_val[pred[i], val_labels[i]] += 1

                    # break

            val_accuracy /= n
            val_loss /= len(test_list)

            time = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
            msg = f'{time} [Epoch: {epoch+1:02}] Valid acc: {val_accuracy:.4f} | loss: {val_loss:.4f}\n\
                {confusion_matrix_val=}\n'.replace('  ', '')
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(msg)
            print(msg)


if __name__ == '__main__':
    main()
