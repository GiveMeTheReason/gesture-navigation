# Change labels in dataset from [1, 14] to [0, 13]

label_filename = './HandGestureDataset_SHREC2017/train_gestures.txt'
out_filename = './HandGestureDataset_SHREC2017/train_gestures_norm.txt'

with open(label_filename, 'r', encoding = 'utf-8') as f:
    with open(out_filename, 'a', encoding = 'utf-8') as o:
        while True:
            gesture, finger, subject, essai, label_14, label_28, seq_len = list(map(int, f.readline().split()))
            o.write(f'{gesture} {finger} {subject} {essai} {label_14 - 1} {label_28 - 1} {seq_len}\n')
