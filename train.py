import torch
from torch.utils.data import DataLoader

from loader import SHREC2017_Dataset
from loss import CrossEntropyLoss
from model import CNN_Classifier
from transforms import Train_Transforms, Test_Transforms

import os
from datetime import datetime 

def main():
    log_filename = 'train_log.txt'
    root_dir = os.path.join(os.path.expanduser("~"), 'datasets/SHREC2017/HandGestureDataset_SHREC2017/')

    train_transforms = Train_Transforms()
    test_transforms = Test_Transforms()

    train_set = SHREC2017_Dataset(root_dir, frames=8, train=True, transform=train_transforms)
    test_set = SHREC2017_Dataset(root_dir, frames=8, train=False, transform=test_transforms)

    train_loader = DataLoader(train_set, batch_size=8, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=8, num_workers=1)

    model = CNN_Classifier(frames=8, num_classes=14)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model.to(device)

    checkpoint_path = "checkpoint.pth"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_func = CrossEntropyLoss()

    epochs = 20

    for epoch in range(epochs):

        model.train()
        train_accuracy = 0
        train_loss = 0
        n = 0

        for images, labels in train_loader:
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
            log_file.write(f'{time} [Epoch: {epoch+1}] Train acc: {train_accuracy} | loss: {train_loss}\n')

        if (epoch+1) % 5 == 0:
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

                    accuracy += (prediction_labels == labels).float().sum()
                    loss += loss_func(prediction, labels).item()
                    n += len(labels)
            
            accuracy /= n
            loss /= len(test_loader)
            time = datetime.now().strftime("%H:%M:%S")

            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(f'{time} [Epoch: {epoch+1}] Val acc: {accuracy} | loss: {loss}\n')

            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

if __name__ == '__main__':
    main()
