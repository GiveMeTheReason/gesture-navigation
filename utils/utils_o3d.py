import typing as tp

import numpy as np

import open3d as o3d

image_sizeT = tp.Tuple[int, int]


def get_visualizer(
    image_size: image_sizeT,
    render_option: str,
) -> o3d.visualization.Visualizer:
    """
    Returns an open3d visualizer with current configuration

    Parameters:
    image_size (image_sizeT): image size to be rendered
    render_option (str): JSON file with rendering options

    Returns:
    o3d.visualization.Visualizer: open3d visualizer
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=image_size[0], height=image_size[1])
    visualizer.get_render_option().load_from_json(render_option)
    return visualizer


def read_image(
    filename: str,
) -> o3d.geometry.Image:
    """
    Returns open3d image from file

    Parameters:
    filename (str): path to read the point cloud

    Returns:
    o3d.geometry.PointCloud: open3d image
    """
    return o3d.io.read_image(filename)


def read_point_cloud(
    filename: str,
    format: str = 'auto',
    remove_nan_points: bool = False,
    remove_infinite_points: bool = False,
    print_progress: bool = False,
) -> o3d.geometry.PointCloud:
    """
    Returns point cloud from file

    Parameters:
    filename (str): path to read the point cloud
    format (str, default='auto'): inferred from file extension
    remove_nan_points (bool, default=False): removes all points that include a NaN
    remove_infinite_points (bool, default=False): removes all points that include a Inf
    print_progress (bool, default=False): progress bar is visualized

    Returns:
    o3d.geometry.PointCloud: point cloud
    """
    return o3d.io.read_point_cloud(
        filename,
        format,
        remove_nan_points,
        remove_infinite_points,
        print_progress,
    )


def write_image(
    filename: str,
    image: tp.Union[o3d.geometry.Image, np.ndarray],
    quality: int = -1,
) -> bool:
    """
    Saves the image into the file. Can accept numpy images (W*H*C)

    Parameters:
    filename (str): path to save the image with extension
    image (o3d.geometry.Image): image to save
    quality (int, default=-1): quality of the output file

    Returns:
    bool: saving result
    """
    return o3d.io.write_image(
        filename,
        o3d.geometry.Image(image),
        quality,
    )


def write_point_cloud(
    filename: str,
    pointcloud: o3d.geometry.PointCloud,
    write_ascii: bool = False,
    compressed: bool = False,
    print_progress: bool = False,
) -> bool:
    """
    Saves the point cloud into the file. Can accept numpy images (W*H*C)

    Parameters:
    filename (str): path to save the point cloud with extension
    pointcloud (o3d.geometry.PointCloud): point cloud to save
    write_ascii (bool, default=False): binary or ascii format
    compressed (bool, default=False): write in compressed format
    print_progress (bool, default=False): progress bar is visualized 

    Returns:
    bool: saving result
    """
    return o3d.io.write_point_cloud(
        filename,
        pointcloud,
        write_ascii,
        compressed,
        print_progress,
    )


def get_rgbd_images(
    images_paths,
    depth_scale=1000,
    depth_trunc=5.0,
) -> tp.List[o3d.geometry.RGBDImage]:
    """
    Returns a list [N // 2] of open3d.geometry.RGBDImage
    Input list should contain paths in the same order as all pipeline
    N = number of paths

    Parameters:
    images_paths (Sequence[str]): filenames for color and depth images
    depth_scale (float, default=1000.0): ratio to scale depth
    depth_trunc (float, default=5.0): values larger depth_trunc gets truncated.
        The depth values will first be scaled and then truncated.

    Returns:
    rgbd_images (list[open3d.geometry.RGBDImage]): RGB-D images
    """
    assert all(images_paths)
    assert len(images_paths) % 2 == 0

    rgbd_images = [[] for _ in range(len(images_paths) // 2)]

    for i in range(len(rgbd_images)):
        rgbd_images[i] = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.io.read_image(images_paths[2 * i]),
            depth=o3d.io.read_image(images_paths[2 * i + 1]),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

    return rgbd_images


def create_point_clouds(
    rgbd_images: tp.List[o3d.geometry.RGBDImage],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tp.List[o3d.geometry.PointCloud]:
    """
    Returns a list [N] of open3d.geometry.PointCloud
    Input list should contain RGB-D images as open3d.geometry.RGBDImage
    All inputs should have the same shape
    N = number of images

    Parameters:
    rgbd_images (Sequence[open3d.geometry.RGBDImage]): RGB-D images
    intrinsics (np.ndarray[float]): intrinsics of RGB-D images
    extrinsics (np.ndarray[float]): extrinsics of RGB-D images

    Returns:
    rgbd_images (list[open3d.geometry.RGBDImage]): RGB-D images
    """
    assert len(rgbd_images) == len(intrinsics) == len(extrinsics)
    assert all(rgbd_images)

    point_clouds = [[] for _ in range(len(rgbd_images))]
    to_filter = [1, 1, 0, 0, 0, 0]

    for i in range(len(point_clouds)):
        point_clouds[i] = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_images[i],
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                *map(lambda val: int(val[1]) if to_filter[val[0]] else val[1], enumerate([*intrinsics[i]]))
            ),
            extrinsic=np.linalg.inv(extrinsics[i]),
            project_valid_depth_only=True
        )

    return point_clouds


def concatenate_point_clouds(
    point_clouds: tp.List[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """
    Returns concateneted point cloud from list of point clouds

    Parameters:
    point_clouds (Sequence[open3d.geometry.PointCloud]): list of point clouds

    Returns:
    concatenated_point_cloud (open3d.geometry.PointCloud): concatenated point cloud
    """
    concatenated_point_cloud = np.sum(point_clouds)

    return concatenated_point_cloud


def filter_by_image_size(point_cloud_data, width, height):
    """
    Returns an array of points that are inside given image size

    Parameters:
    point_cloud_data (np.ndarray[float]): numpy array representation of point cloud
    width (int): image width
    height (int): image height

    Returns:
    filtered_point_cloud_data (np.ndarray[float]): numpy array representation of point cloud
    """
    point_cloud_index = (
        (0 <= point_cloud_data[:, 0]) *
        (point_cloud_data[:, 0] < width) *
        (0 <= point_cloud_data[:, 1]) * 
        (point_cloud_data[:, 1] < height)
    )
    filtered_point_cloud_data = point_cloud_data[point_cloud_index]

    return filtered_point_cloud_data


def project_point_clouds(point_cloud, K, T, width, height):
    """
    Projects point cloud on image with given camera parameters and image size
    Projection goes on camera shifted by angle around z_target point in YZ-plane
    Returns projected image as numpy array

    Parameters:
    point_cloud (open3d.geometry.PointCloud): point cloud to project
    K (np.ndarray[float]): camera intrinsic matrix [4 x 4]
    T (np.ndarray[float]): camera extrinsic matrix [4 x 4]
    width (int): width of the image
    height (int): height of the image

    Returns:
    image_projected (np.ndarray[float]): projected image
    """
    # Create a copy
    point_cloud = o3d.geometry.PointCloud(point_cloud)

    point_cloud.transform(K @ np.linalg.inv(T))

    point_cloud_data = np.hstack([np.asarray(point_cloud.points), np.asarray(point_cloud.colors)])

    point_cloud_data.view('i8,i8,i8,i8,i8,i8').sort(order=['f2'], axis=0)
    point_cloud_data = point_cloud_data[::-1]
    point_cloud_data[:, :3] /= point_cloud_data[:, 2].reshape(-1, 1)

    filtered_point_cloud_data = filter_by_image_size(point_cloud_data, width, height)

    image_projected = np.zeros((height, width, 3))
    image_projected[filtered_point_cloud_data[:, 1].astype(np.int16),
                filtered_point_cloud_data[:, 0].astype(np.int16)] = filtered_point_cloud_data[:, 3:]
    
    return image_projected
