import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model.model_cnn import CNNClassifier

PARTICIPANT = 'G103'
GESTURE = 'start'
HAND = 'right'
TRIAL = 'trial5'

BASE_DIR = os.path.join(
    'D:\\',
    'GesturesNavigation',
    'dataset',
)

TEST_SAMPLE_PATH = os.path.join(
    BASE_DIR,
    PARTICIPANT,
    GESTURE,
    HAND,
    TRIAL,
)

GESTURES_SET = (
    # 'high',
    'start',
    'select',
    # 'swipe_right',
    # 'swipe_left',
)

resized_image_size = (72, 128)
frames = 1
base_fps = 30
target_fps = 5
# target_fps = 30

label_map = {gesture: i for i, gesture in enumerate(GESTURES_SET, start=1)}
label_map['no_gesture'] = 0

batch_size = 8

CHECHPOINT_PATH = 'checkpoint01.pth'


model = CNNClassifier(
    resized_image_size,
    frames=frames,
    batch_size=batch_size,
    num_classes=len(label_map),
)
torch_device = 'cpu' if not torch.cuda.is_available() else 'cuda'
model.to(torch_device)
model.load_state_dict(torch.load(CHECHPOINT_PATH, map_location=torch_device))
model.eval()


to_tensor = transforms.ToTensor()

resize = transforms.Resize(resized_image_size, transforms.InterpolationMode.NEAREST)

class NormalizeDepth():
    def __call__(self, tensor):
        mask = tensor > 0
        tensor_min = tensor[mask].min()
        tensor_max = tensor.max()
        res_tensor = torch.zeros_like(tensor)
        res_tensor[mask] = 1 - (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
        return res_tensor

depth_norm = NormalizeDepth()

inv_map = {v: k for k, v in label_map.items()}

rgb_paths = sorted(glob.glob(os.path.join(TEST_SAMPLE_PATH, '*.jpg')))
depth_paths = sorted(glob.glob(os.path.join(TEST_SAMPLE_PATH, '*.png')))

for rgb_path, depth_path in zip(rgb_paths, depth_paths):
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path).astype(np.float32)[:, :, 0]
    depth = np.where(depth > 2000, 0, depth)

    input_image = resize(torch.cat((to_tensor(rgb[:, :, [2, 1, 0]]), depth_norm(to_tensor(depth))), 0))[None, ...]
    preds = model(torch.cat([input_image, torch.zeros((batch_size - 1, *input_image.shape[1:]))]))[0]

    label = inv_map[torch.argmax(preds).item()]

    # if capture.depth.data is not None:
    #     cv2.imshow('Depth', capture.depth.data)
    # if capture.ir.data is not None:
        # cv2.imshow('IR', capture.ir.data)
    if rgb is not None:
        cv2.putText(rgb, label, (500,500), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
        cv2.imshow('Color', rgb)
    # if capture.transformed_depth is not None:
    #     cv2.imshow('Transformed Depth', capture.transformed_depth)
    # if capture.transformed_color is not None:
    #     cv2.imshow('Transformed Color', capture.transformed_color)
    # if capture.transformed_ir is not None:
    #     cv2.imshow('Transformed IR', capture.transformed_ir)

    key = cv2.waitKey(10)
    if key != -1:
        cv2.destroyAllWindows()
        break
