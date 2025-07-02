from model.mobile import MobileNet
import torch.nn as nn
from collections import OrderedDict
import torch
import torchsummary as summary
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        ]
        self.double_conv = nn.Sequential(*layers)

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



class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal ( decoder)
        # x: skip connection ( encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class UNet(nn.Module):
    def __init__(self, n_channels, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = MobileNet_2(n_channels)

        # upsampling с транспонированной сверткой
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv1 = DoubleConv(1024, 512, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv2 = DoubleConv(512, 256, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
    
        self.conv3 = DoubleConv(384, 128, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 64, dropout=dropout)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    # def forward(self, x):
    #     x2, x1, x0 = self.backbone(x)  # encoder features

    #     d1 = self.up1(x0)
    #     x1_att = self.att1(g=d1, x=x1)
    #     d1 = torch.cat([d1, x1_att], dim=1)
    #     d1 = self.conv1(d1)

    #     d2 = self.up2(d1)
    #     x2_att = self.att2(g=d2, x=x2)
    #     d2 = torch.cat([d2, x2_att], dim=1)
    #     d2 = self.conv2(d2)

    #     d3 = self.up3(d2)
    #     # Для третьего skip соединения в MobileNet_2 у вас x2 — возможно, нужно добавить ещё один skip, 
    #     # если нет, можно пропустить attention здесь или использовать d3 как есть
    #     # Здесь предположим, что x2 — последний skip, используем без attention
    #     print(f"d3 shape: {d3.shape}, x2 shape: {x2.shape}")

    #     d3 = torch.cat([d3, x2], dim=1)
    #     d3 = self.conv3(d3)

    #     d4 = self.up4(d3)
    #     d4 = self.conv4(d4)

    #     out = self.out_conv(d4)
    #     return out

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)  # encoder features

        d1 = self.up1(x0)
        x1_att = self.att1(g=d1, x=x1)
        x1_att = F.interpolate(x1_att, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.conv1(d1)

        d2 = self.up2(d1)
        x2_att = self.att2(g=d2, x=x2)
        x2_att = F.interpolate(x2_att, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.conv2(d2)

        d3 = self.up3(d2)
        x2_resized = F.interpolate(x2, size=d3.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = torch.cat([d3, x2_resized], dim=1)
        d3 = self.conv3(d3)

        d4 = self.up4(d3)
        d4 = self.conv4(d4)

        out = self.out_conv(d4)
        return out



# class UNet(nn.Module):
#     def __init__(self, n_channels, num_classes):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.num_classes = num_classes


#         self.backbone = MobileNet_2(n_channels)

#         self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.conv1 = DoubleConv(1024, 512)

#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.conv2 = DoubleConv(1024, 256)

#         self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
#         self.conv3 = DoubleConv(512, 128)

#         self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
  
#         self.conv4 = DoubleConv(128, 64)

#         self.oup = nn.Conv2d(64, num_classes, kernel_size=1)

#     def forward(self, x):
#         #  backbone
#         x2, x1, x0 = self.backbone(x)

#         P5 = self.up1(x0)
#         P5 = self.conv1(P5)        
#         P4 = x1                      
#         P4 = torch.cat([P4, P5], axis=1)   

#         P4 = self.up2(P4)               
#         P4 = self.conv2(P4)             
#         P3 = x2                         
#         P3 = torch.cat([P4, P3], axis=1)  

#         P3 = self.up3(P3)
#         P3 = self.conv3(P3)

#         P3 = self.up4(P3)
#         P3 = self.conv4(P3)

#         out = self.oup(P3)
       
#         return out
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, num_classes=1)
    model.to(device)
    summary(model, input_size=(3, 256, 512))
