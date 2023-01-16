import os
import typing as tp

import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as T
from PIL import Image

import utils.utils as utils

image_sizeT = tp.Tuple[int, int]
nearest = T.InterpolationMode.NEAREST

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TrainRGBTransforms():
    def __init__(
        self,
        target_size: image_sizeT,
        mean: tp.List[float] = MEAN,
        std: tp.List[float] = STD,
    ) -> None:
        self.transforms = T.Compose([
            T.Resize(
                target_size,
                interpolation=nearest,
            ),
            T.Normalize(
                mean=mean,
                std=std,
            ),
            # T.RandomHorizontalFlip(
            #     p=0.5,
            # ),
            # T.RandomAffine(
            #     degrees=30,
            #     translate=(0.3, 0.3),
            #     scale=(0.7, 1.3),
            #     interpolation=nearest,
            # ),
            # T.RandomPerspective(
            #     distortion_scale=0.4,
            #     p=0.5,
            #     interpolation=nearest,
            # ),
        ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)


class TrainDepthTransforms():
    def __init__(
        self,
        target_size: image_sizeT,
        with_inverse: bool = False,
    ) -> None:
        self.transforms = T.Compose([
            T.Resize(
                target_size,
                interpolation=nearest,
            ),
            NormalizeDepth(
                with_inverse=with_inverse,
            ),
        ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)


class TestRGBTransforms():
    def __init__(
        self,
        target_size: image_sizeT,
        mean: tp.List[float] = MEAN,
        std: tp.List[float] = STD,
    ) -> None:
        self.transforms = T.Compose([
            T.Resize(
                target_size,
                interpolation=nearest,
            ),
            T.Normalize(
                mean=mean,
                std=std,
            ),
        ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)


class TestDepthTransforms():
    def __init__(
        self,
        target_size: image_sizeT,
        with_inverse: bool = False,
    ) -> None:
        self.transforms = T.Compose([
            T.Resize(
                target_size,
                interpolation=nearest,
            ),
            NormalizeDepth(
                with_inverse=with_inverse,
            ),
        ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)


class NormalizeDepth():
    def __init__(self, with_inverse: bool = False) -> None:
        self.with_inverse = with_inverse

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 3 and tensor.shape[0] == 4:
            depth_tensor = tensor[3:]
        else:
            depth_tensor = tensor

        mask = depth_tensor > 0
        tensor_min = depth_tensor[mask].min()
        tensor_max = depth_tensor.max()
        if self.with_inverse:
            depth_tensor[mask] = 1 - (depth_tensor[mask] -
                                      tensor_min) / (tensor_max - tensor_min)
        else:
            depth_tensor[mask] = (depth_tensor[mask] -
                                  tensor_min) / (tensor_max - tensor_min)
        return tensor


class PointCloudToRGBD():
    def __init__(
        self,
        batch_size: int,
        intrinsics: tp.List[float],
        visualizer: o3d.visualization.Visualizer,
        angle: float = 0.0,
        z_target: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
        rgb_transforms: tp.Optional[T.Compose] = None,
        depth_transforms: tp.Optional[T.Compose] = None,
    ) -> None:
        self.rgb_transforms = rgb_transforms
        self.depth_transforms = depth_transforms

        self.intrinsics = intrinsics
        self.visualizer = visualizer

        self.loc = loc
        self.scale = scale

        self.prev_path = [''] * batch_size

        self.base_extrinsic_matrix = utils.build_extrinsic_matrix(
            angle, z_target)
        self.extrinsics = [utils.randomize_extrinsic(
            self.base_extrinsic_matrix, self.loc, self.scale
        ) for _ in range(batch_size)]

        self.camera = o3d.camera.PinholeCameraParameters()
        self.camera.extrinsic = np.linalg.inv(self.base_extrinsic_matrix)

        to_filter = [1, 1, 0, 0, 0, 0]
        self.camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *map(lambda val: int(val[1]) if to_filter[val[0]] else val[1],
                 enumerate([*intrinsics]))
        )

        self.view_control = self.visualizer.get_view_control()

    def _refresh_extrinsic(self, pc_path: str, batch_idx: int) -> None:
        if os.path.dirname(pc_path) != self.prev_path[batch_idx]:
            self.prev_path[batch_idx] = os.path.dirname(pc_path)
            self.extrinsics[batch_idx] = utils.randomize_extrinsic(
                self.base_extrinsic_matrix, self.loc, self.scale)

    def _get_rgb(self, pc: o3d.geometry.PointCloud, batch_idx: int) -> torch.Tensor:
        self.camera.extrinsic = self.extrinsics[batch_idx]
        self.visualizer.add_geometry(pc)
        # self.visualizer.update_geometry()
        self.view_control.convert_from_pinhole_camera_parameters(
            self.camera, True)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        rendered_image = torch.from_numpy(
            np.asarray(
                self.visualizer.capture_screen_float_buffer(do_render=True)
            )
        ).permute(2, 0, 1)
        self.visualizer.remove_geometry(pc)
        self.visualizer.clear_geometries()

        return rendered_image

    def _get_depth(self, pc: o3d.geometry.PointCloud) -> torch.Tensor:
        *image_size, = map(int, self.intrinsics[1::-1])
        K = np.eye(4)
        K[:3, :3] = self.camera.intrinsic.intrinsic_matrix
        T = self.camera.extrinsic

        depth_points = np.asarray(pc.transform(T).points)
        points_n = depth_points.shape[0]
        depth_points = np.hstack([depth_points, np.ones((points_n, 1))])

        depth_projected = depth_points @ K.T
        depth_projected = depth_projected / \
            depth_points[:, 2].reshape(points_n, 1)
        depth_projected = np.delete(depth_projected, 2, axis=1)
        depth_projected[:, 2] = 1 / depth_projected[:, 2]
        depth_projected[:, :2] = depth_projected[:, :2].round()

        depth_index = (
            (0 <= depth_projected[:, 0]) *
            (depth_projected[:, 0] < image_size[1]) *
            (0 <= depth_projected[:, 1]) *
            (depth_projected[:, 1] < image_size[0])
        )

        depth_projected = depth_projected[depth_index]

        depth_on_image = np.full((image_size[0], image_size[1]), 0.)
        depth_on_image[
            depth_projected[:, 1].astype(np.int16),
            depth_projected[:, 0].astype(np.int16)] = depth_projected[:, 2]

        depth_on_image = torch.unsqueeze(torch.from_numpy(depth_on_image), 0)

        return depth_on_image

    def __call__(self, pc_path: str, batch_idx: int) -> torch.Tensor:
        self._refresh_extrinsic(pc_path, batch_idx)

        pc = o3d.io.read_point_cloud(pc_path)

        image_rgb = self._get_rgb(pc, batch_idx)
        image_depth = self._get_depth(pc)

        # image_background = torch.rand(
        #     (3, *map(int, self.intrinsics[1::-1]))
        # )
        # image_background = torch.zeros((3, *map(int, self.intrinsics[1::-1])))
        # image_rgb = torch.where(image_rgb == torch.zeros(3), image_background, image_rgb)

        if self.rgb_transforms:
            image_rgb = self.rgb_transforms(image_rgb)
        if self.depth_transforms:
            image_depth = self.depth_transforms(image_depth)

        rendered_image = torch.cat((
            image_rgb,
            image_depth,
        ), 0)

        return rendered_image


class RGBDepthToRGBD():
    def __init__(
        self,
        rgb_transforms: tp.Optional[T.Compose] = None,
        depth_transforms: tp.Optional[T.Compose] = None,
    ) -> None:
        self.rgb_transforms = rgb_transforms
        self.depth_transforms = depth_transforms

        self.pil_to_tensor = T.PILToTensor()

    def __call__(self, pc_path: str, batch_idx: int = -1) -> torch.Tensor:
        pc_path_jpg = pc_path
        pc_path_png = os.path.join(os.path.splitext(pc_path)[0] + '.png')

        pil_rgb = Image.open(pc_path_jpg)
        pil_depth = Image.open(pc_path_png)

        image_rgb = self.pil_to_tensor(pil_rgb) / 255
        image_depth = self.pil_to_tensor(
            pil_depth) / 255 * float(pil_depth.text.get('MaxDepth', 1))

        # image_background = torch.rand(
        #     (*map(int, self.intrinsics[1::-1]), 3)
        # )

        if self.rgb_transforms:
            image_rgb = self.rgb_transforms(image_rgb)
        if self.depth_transforms:
            image_depth = self.depth_transforms(image_depth)

        rendered_image = torch.cat((
            image_rgb,
            image_depth,
        ), 0)

        return rendered_image
