import typing as tp

import torch
import torch.nn as nn


class ResNet_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        mode = 'identity'
    ) -> None:
        super().__init__()

        self.activation = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

        if mode == 'identity':
            self.identity_resample = nn.Identity()
        elif mode == 'up':
            self.identity_resample = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode == 'down':
            self.identity_resample = nn.MaxPool2d(kernel_size=2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.bn1(tensor)
        out = self.activation(out)
        out = self.identity_resample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        identity = self.identity_resample(tensor)
        identity = self.identity_conv(identity)
        out = out + identity

        return out


class CNN_Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 32,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ResNet_Block(in_channels, 8, mode='down'),
            ResNet_Block(8, 16, mode='identity'),
            ResNet_Block(16, 16, mode='down'),
            ResNet_Block(16, 32, mode='identity'),
            ResNet_Block(32, out_channels, mode='down'),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.blocks(tensor)


class Linear_Head(nn.Module):
    def __init__(
        self,
        in_dim: int = 2*64*9*16,
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            nn.Flatten(),

            nn.Linear(in_dim, 8 * 9 * 16),
            nn.BatchNorm1d(8 * 9 * 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(8 * 9 * 16, 9 * 16),
            nn.BatchNorm1d(9 * 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(9 * 16, num_classes),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.blocks(tensor)


class CNN_Classifier(nn.Module):
    def __init__(
        self,
        image_size: tp.Tuple[int, int],
        frames: int = 8,
        batch_size: int = 1,
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        self.register_buffer(
            'queue_rgb',
            torch.zeros((batch_size, 3 * frames, *image_size)),
            persistent=False,
        )
        self.register_buffer(
            'queue_depth',
            torch.zeros((batch_size, 1 * frames, *image_size)),
            persistent=False,
        )

        self.cnn_model_rgb = CNN_Model(
            in_channels=3*frames,
            out_channels=32,
        )
        self.cnn_model_depth = CNN_Model(
            in_channels=frames,
            out_channels=32,
        )

        self.head = Linear_Head(
            in_dim=2*32*image_size[0]*image_size[1] // 64,
            num_classes=num_classes,
        )

    def _push_to_tensor_rgb(
        self,
        rgb: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((self.queue_rgb[:, 3:], rgb), dim=1)

    def _push_to_tensor_depth(
        self,
        depth: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((self.queue_depth[:, 1:], depth), dim=1)

    def forward(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        rgb = image[:, :3]
        depth = image[:, 3:]

        self.queue_rgb = self._push_to_tensor_rgb(rgb)
        self.queue_depth = self._push_to_tensor_depth(depth)

        rgb_features = self.cnn_model_rgb(self.queue_rgb)
        depth_features = self.cnn_model_depth(self.queue_depth)

        return self.head(torch.cat((rgb_features, depth_features), dim=1))
