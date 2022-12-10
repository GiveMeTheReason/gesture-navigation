import dataclasses
import typing as tp

import k4a


class KinectInitializationException(Exception):
    pass


@dataclasses.dataclass
class DeviceConfig:
    color_format: k4a.EImageFormat = k4a.EImageFormat.COLOR_BGRA32
    color_resolution: k4a.EColorResolution = k4a.EColorResolution.RES_1080P
    depth_mode: k4a.EDepthMode = k4a.EDepthMode.WFOV_2X2BINNED
    camera_fps: k4a.EFramesPerSecond = k4a.EFramesPerSecond.FPS_15
    synchronized_images_only: bool = True
    depth_delay_off_color_usec: int = 0
    wired_sync_mode: k4a.EWiredSyncMode = k4a.EWiredSyncMode.STANDALONE
    subordinate_delay_off_master_usec: int = 0
    disable_streaming_indicator: bool = False


class AzureKinect():
    def __init__(
        self,
        device_config: tp.Optional[k4a.DeviceConfiguration] = None,
        device_index: int = 0,
    ) -> None:
        if device_config is None:
            device_config = self._get_default_device_config()
        self.device_config = device_config

        self.device = k4a.Device.open(device_index)
        if self.device is None:
            raise KinectInitializationException('Azure Kinect not initialized')

    @staticmethod
    def _get_default_device_config() -> k4a.DeviceConfiguration:
        return k4a.DeviceConfiguration(**dataclasses.asdict(DeviceConfig()))

    @classmethod
    def open(
        cls,
        device_config: tp.Optional[k4a.DeviceConfiguration] = None,
        device_index: int = 0
    ) -> k4a.Device:
        if device_config is None:
            device_config = AzureKinect._get_default_device_config()

        device = k4a.Device.open(device_index)
        if device is None:
            raise KinectInitializationException('Azure Kinect not initialized')
        return device


device = AzureKinect.open()
