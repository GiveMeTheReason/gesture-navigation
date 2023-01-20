import glob
import os

import cv2
import numpy as np
import torch

import model.transforms as transforms
from model.model_cnn import CNNClassifier

PARTICIPANT = 'G104'
GESTURE = 'start'
HAND = 'left'
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
    'start',
    'select',
)

resized_image_size = (72, 128)
frames = 1
base_fps = 30
target_fps = 5
# target_fps = 30

label_map = {gesture: i for i, gesture in enumerate(GESTURES_SET)}
label_map['no_gesture'] = len(label_map)

batch_size = 1

CHECHPOINT_PATH = os.path.join(
    'outputs',
    'checkpoint01.pth',
)


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

depth_tf = transforms.TestDepthTransforms(resized_image_size, True)
rgb_tf = transforms.TestRGBTransforms(resized_image_size)

inv_map = {v: k for k, v in label_map.items()}

rgb_paths = sorted(glob.glob(os.path.join(TEST_SAMPLE_PATH, '*.jpg')))
depth_paths = sorted(glob.glob(os.path.join(TEST_SAMPLE_PATH, '*.png')))

log_filename = f'tst_sample_{PARTICIPANT}_{GESTURE}_{HAND}_{TRIAL}.txt'

for rgb_path, depth_path in zip(rgb_paths, depth_paths):
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path).astype(np.float32)[:, :, 0]
    depth = np.where(depth > 2000, 0, depth)

    with torch.no_grad():
        rgb_tensor = to_tensor(rgb[:, :, [2, 1, 0]])
        # image_background = torch.rand(
        #     (3, *map(int, self.intrinsics[1::-1]))
        # )
        # image_background = torch.zeros_like(rgb_tensor)
        # image_background[0] = 1
        # rgb_tensor = torch.where(rgb_tensor == torch.zeros(3, 1, 1), image_background, rgb_tensor)

        np_bg = np.zeros_like(rgb)
        np_bg[:, :, 0] = 255
        rgb = np.where(rgb == np.zeros(3), np_bg, rgb)

        transformed_rgb = rgb_tf(rgb_tensor)
        transformed_depth = depth_tf(to_tensor(depth))

        input_image = torch.cat(
            (
                transformed_rgb,
                transformed_depth,
            ), 0
        )[None, ...]

        preds = model(input_image)

    label = inv_map[torch.argmax(preds).item()]

    with open(log_filename, 'a', encoding='utf-8') as log_file:
        frame_num = os.path.splitext(os.path.basename(rgb_path))[0]
        log_file.write(
            ''.join([
                f'{gest}: {pred:>9.4f} | ' 
                for gest, pred
                in zip(GESTURES_SET + ('no_gesture',), preds[0])
            ]) + f'{frame_num:>5} | ' + f'{label}\n')

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
