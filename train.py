import torch
from torch.utils.data import DataLoader

from loader import SHREC2017_Dataset
from loss import CrossEntropyLoss
from model import CNN_Classifier
from transforms import Train_Transforms, Test_Transforms

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

exp_id = '03'

def main():
    log_filename = f"experiments/train_log{exp_id}.txt"
    root_dir = os.path.join(os.path.expanduser("~"), "datasets/SHREC2017/HandGestureDataset_SHREC2017/")

    target_size = (72, 96)
    train_transforms = Train_Transforms(target_size=target_size)
    test_transforms = Test_Transforms(target_size=target_size)

    train_set = SHREC2017_Dataset(root_dir, frames=8, train=True, transform=train_transforms)
    test_set = SHREC2017_Dataset(root_dir, frames=8, train=False, transform=test_transforms)

    train_loader = DataLoader(train_set, batch_size=16, num_workers=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=1, shuffle=True)

    model = CNN_Classifier(frames=8, num_classes=14)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model.to(device)

    checkpoint_path = f"experiments/checkpoint{exp_id}.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_func = CrossEntropyLoss()

    epochs = 50
    validate_each_epoch = 5

    time = datetime.now().strftime("%H:%M:%S")
    with open(log_filename, 'a', encoding='utf-8') as log_file:
        log_file.write(f'{time} | Training for {epochs} epochs started!\n')
        log_file.write(f'Train set length: {len(train_set)}\n')
        log_file.write(f'Test set length: {len(test_set)}\n')
    print(f'{time} | Training for {epochs} started!')
    print(f'Train set length: {len(train_set)}')
    print(f'Test set length: {len(test_set)}')

    for epoch in range(epochs):

        model.train()
        train_accuracy = 0
        train_loss = 0
        n = 0

        for images, labels in train_loader:

            # plt.figure()
            # img1 = np.hstack([images[0][0].numpy(), images[0][1].numpy()])
            # img2 = np.hstack([images[0][2].numpy(), images[0][3].numpy()])
            # img3 = np.hstack([images[0][4].numpy(), images[0][5].numpy()])
            # img4 = np.hstack([images[0][6].numpy(), images[0][7].numpy()])
            # img = np.vstack([img1, img2, img3, img4])
            # plt.imsave('sample.png', img)
            # break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            prediction = model(images)
            loss = loss_func(prediction, labels)
            loss.backward()
            optimizer.step()

            prediction_probs, prediction_labels = prediction.max(1)
            train_accuracy += (prediction_labels == labels).float().sum()
            train_loss += loss.item()
            n += len(labels)

        train_accuracy /= n
        train_loss /= len(train_loader)
        time = datetime.now().strftime("%H:%M:%S")

        with open(log_filename, 'a', encoding='utf-8') as log_file:
            log_file.write(f'{time} [Epoch: {epoch+1:02}] Train acc: {train_accuracy:.4f} | loss: {train_loss:.4f}\n')
        print(f'{time} [Epoch: {epoch+1:02}] Train acc: {train_accuracy:.4f} | loss: {train_loss:.4f}')

        if (epoch+1) % validate_each_epoch == 0:
            model.eval()
            accuracy = 0
            loss = 0
            n = 0
            with torch.no_grad():
                for val_images, val_labels in test_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    prediction = model(val_images)
                    prediction_probs, prediction_labels = prediction.max(1)

                    accuracy += (prediction_labels == val_labels).float().sum()
                    loss += loss_func(prediction, val_labels).item()
                    n += len(val_labels)

            accuracy /= n
            loss /= len(test_loader)
            time = datetime.now().strftime("%H:%M:%S")

            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f'{time} [Epoch: {epoch+1:02}] Valid acc: {accuracy:.4f} | loss: {loss:.4f}\n')
            print(f'{time} [Epoch: {epoch+1:02}] Valid acc: {accuracy:.4f} | loss: {loss:.4f}')

            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

if __name__ == '__main__':
    main()
