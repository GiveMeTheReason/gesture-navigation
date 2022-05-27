import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class SHREC2017_Dataset(Dataset):
    def __init__(self, root_dir, frames=8, train=True, transform=None):
        # image size
        self.h, self.w = (480, 640)
        self.frames = frames

        self.root_dir = root_dir
        if train:
            labels_path = os.path.join(self.root_dir, 'train_gestures.txt')
        else:
            labels_path = os.path.join(self.root_dir, 'test_gestures.txt')
        self.labels = np.loadtxt(labels_path, dtype=np.uint8)
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        gesture, finger, subject, essai, label_14, label_28, seq_len = self.labels[index]
        seq_path = os.path.join(self.root_dir,
                                f'gesture_{gesture}/',
                                f'finger_{finger}/',
                                f'subject_{subject}/',
                                f'essai_{essai}/')

        # slice video in self.frames parts
        bins = np.linspace(0, seq_len, self.frames + 1)
        seq = np.zeros((self.frames, self.h, self.w), dtype=np.float32)
        for i in range(self.frames):
            # sample from part
            frame_id = np.random.randint(bins[i], bins[i + 1])
            image_path = os.path.join(seq_path, f'{frame_id}_depth.png')
            seq[i] = np.array(Image.open(image_path), dtype=np.float32)
        
        # apply transforms
        if self.transform:
            seq = self.transform(torch.from_numpy(seq))
        return seq, label_14
