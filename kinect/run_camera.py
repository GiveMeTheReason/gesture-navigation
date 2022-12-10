import contextlib
import sys

import cv2
import k4a
import numpy as np
import plotly
import plotly.graph_objs as go


def viz(device: k4a.Device):
    while True:
        capture = device.get_capture(-1)
        if capture.depth.data is not None:
            cv2.imshow('Depth', capture.depth.data)
        if capture.ir.data is not None:
            cv2.imshow('IR', capture.ir.data)
        if capture.color.data is not None:
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


def main():
    with k4a.Device.open() or contextlib.nullcontext() as device:
        if device is None:
            sys.exit(-1)
        run(device)


def run(device: k4a.Device):
    device_config = k4a.DeviceConfiguration(
        color_format=k4a.EImageFormat.COLOR_BGRA32,
        color_resolution=k4a.EColorResolution.RES_720P,
        depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
        camera_fps=k4a.EFramesPerSecond.FPS_30,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False
    )

    # Start Cameras
    status = device.start_cameras(device_config)
    if status != k4a.EStatus.SUCCEEDED:
        sys.exit(-1)

    # Get Calibration
    calibration = device.get_calibration(
        depth_mode=device_config.depth_mode,
        color_resolution=device_config.color_resolution,
    )

    # Create Transformation
    transformation = k4a.Transformation(calibration)

    # Capture One Frame
    capture = device.get_capture(-1)

    # Get Point Cloud
    xyz_image = transformation.depth_image_to_point_cloud(capture.depth, k4a.ECalibrationType.DEPTH)
    xyz = xyz_image.data[np.all(xyz_image.data[:, :] != np.zeros(3), axis=2)]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz_image.data[:, :, 0].reshape(-1),
                y=xyz_image.data[:, :, 1].reshape(-1),
                z=xyz_image.data[:, :, 2].reshape(-1),
                mode='markers',
                # marker=dict(size=1, color=colors_points[::step])
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
    )
    fig.show()

    # viz(device)


if __name__ == '__main__':
    main()
