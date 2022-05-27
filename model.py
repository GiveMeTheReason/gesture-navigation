import torch
import torch.nn as nn


class ResNet_Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        mode = 'identity'
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        if mode == 'identity':
            self.identity_resample = nn.Identity()
        elif mode == 'up':
            self.identity_resample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode == 'down':
            self.identity_resample = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, input: torch.Tensor):
        identity = input

        out = self.bn1(input)
        out = self.activation(out)
        out = self.identity_resample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        identity = self.identity_resample(identity)
        identity = self.identity_conv(identity)
        out += identity

        return out


class CNN_Model(nn.Module):
    def __init__(self, in_channels=8, out_channels=64):
        super().__init__()
        self.blocks = nn.Sequential(
            ResNet_Block(in_channels, 16, mode='down'),
            ResNet_Block(16, 16, mode='identity'),
            ResNet_Block(16, 16, mode='identity'),
            ResNet_Block(16, 32, mode='down'),
            ResNet_Block(32, 32, mode='identity'),
            ResNet_Block(32, 32, mode='identity'),
            ResNet_Block(32, out_channels, mode='down')
            )

    def forward(self, input: torch.Tensor):
        return self.blocks(input)


class Linear_Head(nn.Module):
    def __init__(self, in_dim=64*9*12, num_classes=14):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 8 * 9 * 12),
            nn.ReLU(),

            nn.Dropout(p=0.2),
            nn.BatchNorm1d(8 * 9 * 12),
            nn.Linear(8 * 9 * 12, 9 * 12),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.BatchNorm1d(9 * 12),
            nn.Linear(9 * 12, num_classes),
            )

    def forward(self, input: torch.Tensor):
        return self.blocks(input)


class CNN_Classifier(nn.Module):
    def __init__(self, frames=8, num_classes=14):
        super().__init__()
        self.cnn_model = CNN_Model(in_channels=frames, out_channels=64)
        self.head = Linear_Head(in_dim=64*9*12, num_classes=num_classes)
    
    def forward(self, input: torch.Tensor):
        return self.head(self.cnn_model(input))
