from model.mobile import MobileNet
import torch.nn as nn
from collections import OrderedDict
import torch
import torchsummary as summary

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))


class MobileNet_2(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet_2, self).__init__()
        self.model = MobileNet(n_channels)

    def forward(self, x):
        out3 = self.model.layer1(x)
        out4 = self.model.layer2(out3)
        out5 = self.model.layer3(out4)

        return out3, out4, out5



class UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes


        self.backbone = MobileNet_2(n_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
  
        self.conv4 = DoubleConv(128, 64)

        self.oup = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        P5 = self.up1(x0)
        P5 = self.conv1(P5)        
        P4 = x1                      
        P4 = torch.cat([P4, P5], axis=1)   

        P4 = self.up2(P4)               
        P4 = self.conv2(P4)             
        P3 = x2                         
        P3 = torch.cat([P4, P3], axis=1)  

        P3 = self.up3(P3)
        P3 = self.conv3(P3)

        P3 = self.up4(P3)
        P3 = self.conv4(P3)

        out = self.oup(P3)
       
        return out
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, num_classes=1)
    model.to(device)
    summary(model, input_size=(3, 256, 512))
