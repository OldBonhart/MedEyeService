import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
import matplotlib as mpl
from matplotlib import cm


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))

class UNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)  # ,1

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        out = torch.sigmoid(out)

        return out


unet = UNet(n_classes=1).cpu()
weights = open('blood_vessels_segmentation/unet.pt', 'rb')
unet.load_state_dict(torch.load(weights, map_location='cpu'))
unet = unet.cpu()


def unet_make_predict(img_tensor):
    mask = unet(img_tensor)
    mask = mask.cpu().detach().numpy()
    mask = mask[0, 0, :, :]

    cm_hot = mpl.cm.get_cmap('binary')  # color map
    mask = cm_hot(mask)

    mask = np.uint8(mask * 255)
    mask = Image.fromarray(mask).convert("RGB")
    mask = mask.resize((312, 312), Image.BILINEAR)
    return mask
