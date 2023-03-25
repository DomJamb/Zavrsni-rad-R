import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from ResidualBlock import ResidualBlock

class ResidualNetwork18(nn.Module):
    """
    Class modeling the ResNet18 (1 convolutional layer, 4 residual layers, 1 linear layer)
    """   
          
    def __init__(self, no_of_classes=10):
        """
        Init function for a ResNet18 net
        Params:
            no_of_classes: number of output classes
        """
                
        super(ResidualNetwork18, self).__init__()

        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)

        # 4 residual layers consisting of 2 residual blocks each
        self.res1 = self.create_residual_layer(in_channels=64, out_channels=64, stride=1)
        self.res2 = self.create_residual_layer(in_channels=64, out_channels=128)
        self.res3 = self.create_residual_layer(in_channels=128, out_channels=256)
        self.res4 = self.create_residual_layer(in_channels=256, out_channels=512)

        # Final linear layer
        self.linear = nn.Linear(512, no_of_classes)

        # Model parameters initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=1e-3)
                init.constant_(module.bias, 0)

    def create_residual_layer(self, in_channels, out_channels, no_of_blocks=2, stride=2):
        """
        Function for creating a residual layer using multiple residual blocks 
        (default is 2 blocks for ResNet18)
        """

        # Strides per layer
        stride_per_layer = [1 for x in range(no_of_blocks)]
        stride_per_layer[0] = stride
        blocks = []

        # Creation of residual blocks with suitable input and output sizes
        for stride in stride_per_layer:
            blocks.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels

        return nn.Sequential(*blocks)

    def forward(self, x):
        """
        Forward function for residual blocks
        Params:
            x: input tensor
        """
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = F.avg_pool2d(y, 4)
        y = y.view(y.size(0), -1)
        y = self.linear(y)
        return y