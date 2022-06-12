# Visualize losses and accuracy stored in train_log.txt file

import re
import matplotlib.pyplot as plt

exp_id = '03'

log_filename = f"experiments/train_log{exp_id}.txt"

train_loss = [0]
train_acc = [0]
valid_loss = [0]
valid_acc = [0]

train_epoch = [0]
valid_epoch = [0]

with open(log_filename, 'r', encoding='utf-8') as log_file:
    line = log_file.readline()
    while line:
        args = re.sub('[\[\]\|\:\n]', '', line).split()

        if len(args) == 8:
            if args[3] == 'Train':
                if int(args[2]) > train_epoch[-1]:
                    train_epoch.append(int(args[2]))
                    train_acc.append(float(args[5]))
                    train_loss.append(float(args[7]))
                else:
                    train_epoch = [0, int(args[2])]
                    train_acc = [0, float(args[5])]
                    train_loss = [0, float(args[7])]
            elif args[3] == 'Valid':
                if int(args[2]) > valid_epoch[-1]:
                    valid_epoch.append(int(args[2]))
                    valid_acc.append(float(args[5]))
                    valid_loss.append(float(args[7]))
                else:
                    valid_epoch = [0, int(args[2])]
                    valid_acc = [0, float(args[5])]
                    valid_loss = [0, float(args[7])]

        line = log_file.readline()

train_loss[0] = train_loss[1]
valid_loss[0] = valid_loss[1]

plt.figure()
plt.plot(train_epoch, train_acc, label='Train accuracy')
plt.plot(valid_epoch, valid_acc, label='Val accuracy')
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(f'experiments/accuracy{exp_id}.png')

plt.figure()
plt.plot(train_epoch, train_loss, label='Train loss')
plt.plot(valid_epoch, valid_loss, label='Val loss')
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('CrossEntropyLoss')
plt.savefig(f'experiments/loss{exp_id}.png')
