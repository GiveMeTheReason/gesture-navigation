import datetime

import torch


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

    msg = 'Training finished!\n'
    with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(msg)
    print(msg)
