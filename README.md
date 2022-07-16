# gestures_navigation

## Requirements

#### Open3D

Augmentations use headless rendering which requeries building **Open3D with OSMesa**

Check the [guide](http://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html)

#### PyTorch3d

**PyTorch3d** is not needed for augmentations but some experiments was performed using it (**renderer** folder)

Installation [guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

Or use instructure below for Python3.8:

1. Install **torch** and **torchvision**

`pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`

2. Install requirements for **pytorch3d**

`pip install -U fvcore`

`pip install -U iopath`

3. Install **pytorch3d**

`pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html`
