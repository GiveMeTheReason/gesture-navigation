import cv2
import k4a
import numpy as np
import torch
import torchvision.transforms as transforms

from model.model_cnn import CNN_Classifier

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


model = CNN_Classifier(
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

while True:
    capture = device.get_capture(-1)

    depth_on_color = transformation.depth_image_to_color_camera(capture.depth)

    rgb = capture.color.data[:, :, :3][:, :, [2, 1, 0]]
    depth = depth_on_color.data.astype(np.float32)
    depth = np.where(depth > 2000, 0, depth)

    input_image = resize(torch.cat((to_tensor(rgb), depth_norm(to_tensor(depth))), 0))[None, ...]
    preds = model(torch.cat([input_image, torch.zeros((batch_size - 1, *input_image.shape[1:]))]))[0]

    label = inv_map[torch.argmax(preds).item()]
    print(preds, capture.color.device_timestamp_usec)

    # if capture.depth.data is not None:
    #     cv2.imshow('Depth', capture.depth.data)
    # if capture.ir.data is not None:
        # cv2.imshow('IR', capture.ir.data)
    if capture.color.data is not None:
        cv2.putText(capture.color.data, label, (500,500), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
        cv2.imshow('Color', capture.color.data)
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
