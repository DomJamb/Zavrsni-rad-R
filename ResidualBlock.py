import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Class representing a residual block (has 1 skip connection)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Init function for residual blocks
        Params:
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride for first convolutional layer kernel
        """
        super(ResidualBlock, self).__init__()

        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        """
        Forward function for residual blocks
        Params:
            x: input tensor
        """
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y += self.skip(x)
        y = F.relu(y)
        return y