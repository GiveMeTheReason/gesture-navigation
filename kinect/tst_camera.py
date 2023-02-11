import os

import cv2
import k4a
import numpy as np
import torch

import model.transforms as transforms
from config import CONFIG
from model.model_cnn import CNNClassifier

device = k4a.Device.open()
if device is None:
    exit(-1)

device_config = k4a.DeviceConfiguration(
    color_format=k4a.EImageFormat.COLOR_BGRA32,
    color_resolution=k4a.EColorResolution.RES_720P,
    depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
    camera_fps=k4a.EFramesPerSecond.FPS_5,
    synchronized_images_only=True,
    depth_delay_off_color_usec=0,
    wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
    subordinate_delay_off_master_usec=0,
    disable_streaming_indicator=False
)

# Start Cameras
status = device.start_cameras(device_config)
if status != k4a.EStatus.SUCCEEDED:
    exit(-1)

# Get Calibration
calibration = device.get_calibration(
    depth_mode=device_config.depth_mode,
    color_resolution=device_config.color_resolution,
)

# Create Transformation
transformation = k4a.Transformation(calibration)

GESTURES_MAP = CONFIG['gestures']['gestures_set']
GESTURES_SET = [gesture[0] for gesture in GESTURES_MAP]

resized_image_size = CONFIG['train']['resized_image_size']
frames = CONFIG['train']['frames_buffer']
base_fps = CONFIG['train']['base_fps']
target_fps = CONFIG['train']['target_fps']

label_map = {**GESTURES_MAP}
if CONFIG['gestures']['with_rejection']:
    label_map['no_gesture'] = len(label_map)

batch_size = 1

CHECHPOINT_PATH = CONFIG['directories']['outputs']['checkpoint_path']


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

while True:
    capture = device.get_capture(-1)

    depth_on_color = transformation.depth_image_to_color_camera(capture.depth)

    depth = depth_on_color.data.astype(np.float32)
    depth = np.where(depth > 2000, 0, depth)
    rgb = capture.color.data[:, :, :3][:, :, [2, 1, 0]]
    rgb = np.where(depth[..., None] == 0, 0, rgb)

    with torch.no_grad():
        transformed_rgb = rgb_tf(to_tensor(rgb[:, :, [2, 1, 0]]))
        transformed_depth = depth_tf(to_tensor(depth))

        input_image = torch.cat(
            (
                transformed_rgb,
                transformed_depth,
            ), 0
        )[None, ...]

        preds = model(input_image)

    label = inv_map[torch.argmax(preds).item()]
    print(preds, capture.color.device_timestamp_usec)

    with open('tst_camera.txt', 'a', encoding='utf-8') as log_file:
        timestamp = capture.color.device_timestamp_usec
        log_file.write(
            ''.join([
                f'{gest}: {pred:>9.4f} | '
                for gest, pred
                in zip(GESTURES_SET + ['no_gesture'], preds[0])
            ]) + f'{timestamp:>10} | ' + f'{label}\n')

    # if capture.color.depth is not None:
    #     cv2.imshow('Depth', capture.color.depth)
    # if depth is not None:
    #     cv2.putText(depth, label, (500, 500), cv2.FONT_HERSHEY_PLAIN, 3, 0, 5)
    #     cv2.imshow('Depth', depth)
    # if capture.ir.data is not None:
        # cv2.imshow('IR', capture.ir.data)
    # if capture.color.data is not None:
    #     cv2.putText(capture.color.data, label, (500, 500), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    #     cv2.imshow('Color', capture.color.data)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb is not None:
        cv2.putText(rgb, label, (500, 500),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
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
