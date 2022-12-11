import os
import typing as tp

import mrob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import open3d as o3d

image_sizeT = tp.Tuple[int, int]
nearest = transforms.InterpolationMode.NEAREST


class NormalizeDepth():
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor > 0
        tensor_min = tensor[mask].min()
        tensor_max = tensor.max()
        res_tensor = torch.zeros_like(tensor)
        res_tensor[mask] = 1 - (tensor[mask] - tensor_min) / (tensor_max - tensor_min)
        return res_tensor


class Train_Transforms():
    def __init__(
        self,
        target_size: image_sizeT = (72, 96),
    ) -> None:
        self.transforms = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize(
                    target_size,
                    interpolation=nearest,
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5,
                ),
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.3, 0.3),
                    scale=(0.7, 1.3),
                    interpolation=nearest,
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.4,
                    p=0.5,
                    interpolation=nearest,
                ),
                NormalizeDepth()
            ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)


class Test_Transforms():
    def __init__(
        self,
        target_size: image_sizeT = (72, 96),
    ) -> None:
        self.transforms = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Resize(
                    target_size,
                    interpolation=nearest,
                ),
                NormalizeDepth()
            ])

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)


class PointCloud_To_RGBD():
    def __init__(
        self,
        batch_size: int,
        intrinsics: tp.List[float],
        visualizer: o3d.visualization.Visualizer,
        render_option_path: str,
        angle: float = 0.0,
        z_target: float = 1.0,
        loc: float = 0.0,
        scale: float = 1.0,
        image_size: tp.Optional[image_sizeT] = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.visualizer = visualizer
        self.render_option_path = render_option_path

        self.angle = angle
        self.z_target = z_target
        self.loc = loc
        self.scale = scale

        self.resize = transforms.Resize(image_size, transforms.InterpolationMode.NEAREST) if image_size is not None else image_size

        self.intrinsic_matrix = self.get_intrinsic_matrix(*self.intrinsics[2:])

        self.base_extrinsic_matrix = self.get_extrinsic_matrix(self.angle, self.z_target)
        self.extrinsic_matrix = self.base_extrinsic_matrix

        self.camera = o3d.camera.PinholeCameraParameters()

        self.camera.extrinsic = np.linalg.inv(self.extrinsic_matrix)

        self.extrinsics = [self.randomize_extrinsic(
                self.base_extrinsic_matrix, self.loc, self.scale
            ) for _ in range(batch_size)]
        self.prev_path = [''] * batch_size

        to_filter = [1, 1, 0, 0, 0, 0]
        self.camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *map(lambda val: int(val[1]) if to_filter[val[0]] else val[1],
                enumerate([*intrinsics]))
        )

        self.view_control = self.visualizer.get_view_control()

    def get_intrinsic_matrix(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """
        Returns an intrinsic matrix from list of parameters

        Parameters:
        fx (float): focal x in pixels
        fy (float): focal y in pixels
        cx (float): principal point x in pixels
        cy (float): principal point y in pixels

        Returns:
        intrinsic_matrix (np.ndarray[float]): intrinsic matrix
        """
        intrinsic_matrix = np.array([
            [fx, 0., cx, 0.],
            [0., fy, cy, 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float64)

        return intrinsic_matrix

    def get_extrinsic_matrix(
        self,
        angle: float,
        z_target: float,
    ) -> np.ndarray:
        """
        Returns an extrinsic matrix for new camera position from list of parameters
        Camera position calculates as rotation around X-axis shifted by Z on z_target
        So, camera rotates around z_target point

        Parameters:
        angle (float): angle of rotation in radians
        z_target (float): z-coornate of point of rotation

        Returns:
        extrinsic_matrix (np.ndarray[float]): extrinsic matrix
        """
        extrinsic_matrix = np.array([
            [1., 0., 0., 0.],
            [0., np.cos(angle), -np.sin(angle), z_target * np.sin(angle)],
            [0., np.sin(angle), np.cos(angle), z_target * (1 - np.cos(angle))],
            [0., 0., 0., 1.],
        ], dtype=np.float64)

        return extrinsic_matrix

    def randomize_extrinsic(
        self,
        extrinsic_matrix: np.ndarray,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Returns matrix with applied random normal (mean=0, var=var)
        variations on state vector [th1, th2, th3, x, y, z] using mrob

        Parameters:
        extrinsic_matrix (np.ndarray[float]): extrinsic parameters of camera [4 x 4]
        loc (float or Sequence[float]): mean, can be array of 6 floats
        scale (float or Sequence[float]): standard deviation, can be array of 6 floats

        Returns:
        randomized_extrinsic_matrix (np.ndarray[float]): randomized extrinsic matrix
        """
        T = mrob.geometry.SE3(extrinsic_matrix)

        randomized_vector = np.random.normal(loc=loc, scale=scale)
        randomized_extrinsic_matrix = (mrob.geometry.SE3(randomized_vector) * T).T()

        return randomized_extrinsic_matrix

    @property
    def refresh(self) -> None:
        self.extrinsic_matrix = self.randomize_extrinsic(
            self.base_extrinsic_matrix, self.loc, self.scale
        )

        self.camera.extrinsic = np.linalg.inv(self.extrinsic_matrix)

    def __call__(self, pc_path: str, batch_idx: int) -> torch.Tensor:
        if os.path.dirname(pc_path) != self.prev_path[batch_idx]:
            self.prev_path[batch_idx] = os.path.dirname(pc_path)
            self.extrinsics[batch_idx] = self.randomize_extrinsic(
                self.base_extrinsic_matrix, self.loc, self.scale)

        pc = o3d.io.read_point_cloud(pc_path)

        self.camera.extrinsic = self.extrinsics[batch_idx]

        self.visualizer.add_geometry(pc)

        self.view_control.convert_from_pinhole_camera_parameters(self.camera, True)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        rendered_image = torch.from_numpy(
                np.asarray(
                    self.visualizer.capture_screen_float_buffer(do_render=True)
                )
            )
        self.visualizer.clear_geometries()


        # image_background = torch.rand(
        #     (*map(int, self.intrinsics[1::-1]), 3)
        # )
        image_background = torch.zeros((*map(int, self.intrinsics[1::-1]), 3))
        rendered_image = torch.where(rendered_image == torch.zeros(3), image_background, rendered_image).permute(2, 0, 1)


        image_size = (720, 1280)
        K = np.eye(4)
        K[:3, :3] = self.camera.intrinsic.intrinsic_matrix
        T = self.camera.extrinsic

        depth_points = np.asarray(pc.transform(T).points)
        points_n = depth_points.shape[0]
        depth_points = np.hstack([depth_points, np.ones((points_n, 1))])

        depth_projected = depth_points @ K.T
        depth_projected = depth_projected / depth_points[:, 2].reshape(points_n, 1)
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

        mask = depth_on_image > 1.0
        tensor_min = depth_on_image[mask].min()
        tensor_max = depth_on_image.max()
        depth_on_image[mask] = 1 - (depth_on_image[mask] - tensor_min) / (tensor_max - tensor_min)

        depth_on_image = torch.unsqueeze(torch.from_numpy(depth_on_image), 0)


        # NOTE: COMMENT AFTER GENERATING PROXY DATASET
        # o3d.io.write_image(
        #     os.path.splitext(pc_path)[0] + '.jpg',
        #     o3d.geometry.Image((rendered_image * 255).permute(1, 2, 0).numpy().astype(np.int8))
        # )
        # o3d.io.write_image(
        #     os.path.splitext(pc_path)[0] + '.png',
        #     o3d.geometry.Image((depth_on_image * 255).permute(1, 2, 0).numpy().astype(np.int8))
        # )



        rendered_image = torch.cat((rendered_image, depth_on_image), 0)

        if self.resize is not None:
            rendered_image = self.resize(rendered_image)

        return rendered_image


class RGB_Depth_To_RGBD():
    def __init__(
        self,
        image_size: tp.Optional[image_sizeT] = None,
    ) -> None:
        self.resize = transforms.Resize(
            image_size, transforms.InterpolationMode.NEAREST
        ) if image_size is not None else image_size

    def __call__(self, pc_path: str, batch_idx: int) -> torch.Tensor:
        pc_path_jpg = pc_path
        pc_path_png = os.path.join(os.path.splitext(pc_path)[0] + ".png")

        image_rgb = o3d.io.read_image(pc_path_jpg)
        image_depth = o3d.io.read_image(pc_path_png)

        # image_background = torch.rand(
        #     (*map(int, self.intrinsics[1::-1]), 3)
        # )

        rendered_image = torch.cat(
            (
                torch.from_numpy(np.asarray(image_rgb)).permute(2, 0, 1),
                torch.unsqueeze(torch.from_numpy(np.asarray(image_depth)), 0),
            ),
            0,
        )

        if self.resize is not None:
            rendered_image = self.resize(rendered_image)

        return rendered_image
