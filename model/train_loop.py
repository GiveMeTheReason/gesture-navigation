import datetime

import torch

import wandb


def now():
    return datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')


def train_model(
    model,
    train_loader,
    test_loader,
    train_list,
    test_list,
    label_map,
    optimizer,
    loss_func,
    epochs = 1,
    validate_each_epoch = 1,
    target_fps = 30,
    base_fps = 30,
    checkpoint_path = 'checkpoint.pth',
    log_filename = 'train_log.txt',
    device = 'cpu' if not torch.cuda.is_available() else 'cuda',
):
    def log_msg(
        msg: str,
        to_terminal: bool = False,
        to_log_file: bool = False,
    ):
        if to_log_file:
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(msg)
        if to_terminal:
            print(msg)

    wandb.init(project='gestures-navigation')
    wandb.config = {
        'learning_rate': 1e-4,
        'epochs': epochs,
        'batch_size': 2
    }
    wandb.watch(model)

    msg = (
        f'\n{now()} | Training for {epochs} epochs started!\n'
        f'Train set length: {len(train_list)}\n'
        f'Test set length: {len(test_list)}\n'
    )
    log_msg(msg, to_terminal=True, to_log_file=True)

    for epoch in range(epochs):

        model.train()
        train_accuracy = 0
        train_loss = 0
        n = 0

        confusion_matrix_train = torch.zeros((len(label_map), len(label_map)), dtype=torch.int)

        for counter, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            prediction = model(images)
            batch_loss = loss_func(prediction, labels)
            batch_loss.backward()
            optimizer.step()

            msg = (
                f'{now()} TRAIN\n'
                f'{epoch=}, {counter=}/{len(train_list)*120*target_fps//base_fps}\n'
                f'{prediction=}\n'
                f'{labels=}'
            )
            log_msg(msg, to_terminal=True, to_log_file=False)

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

        msg = (
            f'{now()} [Epoch: {epoch+1:02}] Train acc: {train_accuracy:.4f} | loss: {train_loss:.4f}\n'
            f'{confusion_matrix_train=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        torch.save(model.state_dict(), checkpoint_path)

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            **{
                f'train_acc_{str(i).zfill(2)}':
                confusion_matrix_train[cls, cls] / confusion_matrix_train[:, cls].sum()
                for cls in range(len(label_map))
            }
        })

        if not ((epoch+1) % validate_each_epoch == 0):
            continue

        model.eval()
        val_accuracy = 0
        val_loss = 0
        n = 0

        confusion_matrix_val = torch.zeros((len(label_map), len(label_map)), dtype=torch.int)

        with torch.no_grad():
            for counter, (val_images, val_labels) in enumerate(test_loader):

                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                prediction = model(val_images)
                prediction_probs, prediction_labels = prediction.max(1)

                msg = (
                    f'{now()} VAL\n'
                    f'{epoch=}, {counter=}/{len(test_list)*120*target_fps//base_fps}\n'
                    f'{prediction=}\n'
                    f'{val_labels=}'
                )
                log_msg(msg, to_terminal=True, to_log_file=False)

                val_accuracy += (prediction_labels == val_labels).sum().float()
                val_loss += loss_func(prediction, val_labels).item()
                n += len(val_labels)

                pred = torch.argmax(prediction, dim=1)
                for i in range(len(val_labels)):
                    confusion_matrix_val[pred[i], val_labels[i]] += 1

                # break

        val_accuracy /= n
        val_loss /= len(test_list)

        msg = (
            f'{now()} [Epoch: {epoch+1:02}] Valid acc: {val_accuracy:.4f} | loss: {val_loss:.4f}\n'
            f'{confusion_matrix_val=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        wandb.log({
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            **{
                f'val_acc_{str(i).zfill(2)}':
                confusion_matrix_val[cls, cls] / confusion_matrix_val[:, cls].sum()
                for cls in range(len(label_map))
            }
        })

    msg = 'Training finished!\n\n'
    log_msg(msg, to_terminal=True, to_log_file=True)
