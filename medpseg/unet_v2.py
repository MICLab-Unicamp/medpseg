'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

v2
My personal UNet code. Heavily modified from internet code.
A final version with less flexibility since we are not doing experiments on things
that are already well known about UNet configuration

Author: Diedre Carmo
https://github.com/dscarmo

If you use this architecture please cite its original use:
CARMO, Diedre; RITTNER, Leticia; LOTUFO, Roberto. MultiATTUNet: brain tumor segmentation and survival multitasking. In: International MICCAI Brainlesion Workshop. Springer, Cham, 2020. p. 424-434.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom


def assert_dim(dim):
    assert dim in ('2d', '3d'), "dim {} not supported".format(dim)


class SelfAttention(nn.Module):
    '''
    Spatial attention module, with 1x1 convolutions, idea from
    ASSESSING KNEE OA SEVERITY WITH CNN ATTENTION-BASED END-TO-END ARCHITECTURES
    '''
    def __init__(self, in_ch, dim):
        super().__init__()
        self.first_conv = getattr(nn, f"Conv{dim}")(in_ch, in_ch//2, kernel_size=1, padding=0, stride=1, bias=False)
        self.second_conv = getattr(nn, f"Conv{dim}")(in_ch//2, in_ch//4, kernel_size=1, padding=0, stride=1, bias=False)
        self.third_conv = getattr(nn, f"Conv{dim}")(in_ch//4, 1, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        y = self.first_conv(x)
        y = F.leaky_relu(y, inplace=True)
        y = self.second_conv(y)
        y = F.leaky_relu(y, inplace=True)
        self.att = self.third_conv(y).sigmoid()
        return x*self.att


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm, reduce, dim):
        super().__init__()
        
        if norm == "group":
            norms = [nn.GroupNorm(num_groups=8, num_channels=out_ch) for _ in range(2)]
        elif norm == "instance":
            norms = [getattr(nn, f"InstanceNorm{dim}")(out_ch) for _ in range(2)]
        elif norm:
            norms = [getattr(nn, f"BatchNorm{dim}")(out_ch) for _ in range(2)]
        else:
            norms = [nn.Identity() for _ in range(2)]

        # Default conv block
        self.conv = nn.Sequential(
            getattr(nn, f"Conv{dim}")(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
            norms[0],
            nn.LeakyReLU(inplace=True),
            getattr(nn, f"Conv{dim}")(out_ch, out_ch, kernel_size=3, padding=1, stride=2 if reduce else 1, bias=False),
            norms[1],
            nn.LeakyReLU(inplace=True)
        )

        self.residual_connection = getattr(nn, f"Conv{dim}")(in_ch, out_ch, kernel_size=1, padding=0, stride=2 if reduce else 1, bias=False)

    def forward(self, x):
        y = self.conv(x)
        return y + self.residual_connection(x)
        

class Up(nn.Module):
    '''
    Upsample conv
    '''
    def __init__(self, in_ch, out_ch, norm, dim):
        super().__init__()
        self.dim = dim
        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="trilinear" if dim == '3d' else "bilinear")
        self.conv = DoubleConv(in_ch, out_ch, norm, reduce=False, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Fix upsampling issues with odd shapes, from:
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        if self.dim == '3d':
            diffZ = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, (diffZ // 2, diffZ - diffZ // 2,
                            diffY // 2, diffY  - diffY // 2,
                            diffX // 2, diffX - diffX // 2))
        else:
            x1 = F.pad(x1, (diffY // 2, diffY - diffY // 2,
                            diffX // 2, diffX - diffX // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        '''
        Expected args:
        n_channels, init_channel, norm, dim
        '''
        super().__init__()
        n_channels = kwargs["n_channels"]
        init_channel = kwargs["init_channel"]
        norm = kwargs["norm"]
        dim = kwargs["dim"]

        self.inc = DoubleConv(n_channels, init_channel, norm=norm, reduce=False, dim=dim)
        self.att0 = SelfAttention(init_channel, dim=dim)
        self.down1 = DoubleConv(init_channel, init_channel*2, norm=norm, reduce=True, dim=dim)
        self.att1 = SelfAttention(init_channel*2, dim=dim)
        self.down2 = DoubleConv(init_channel*2, init_channel*4, norm=norm, reduce=True, dim=dim)
        self.att2 = SelfAttention(init_channel*4, dim=dim)
        self.down3 = DoubleConv(init_channel*4, init_channel*8, norm=norm, reduce=True, dim=dim)
        self.att3 = SelfAttention(init_channel*8, dim=dim)
        self.down4 = DoubleConv(init_channel*8, init_channel*8, norm=norm, reduce=True, dim=dim)

    def forward(self, x):
        self.input_shape = (x.shape[-3], x.shape[-2], x.shape[-1])
        out_1 = self.att0(self.inc(x))
        out_2 = self.att1(self.down1(out_1))
        out_3 = self.att2(self.down2(out_2))
        out_4 = self.att3(self.down3(out_3))
        y = self.down4(out_4)

        return y, out_1, out_2, out_3, out_4

    def return_atts(self):
        '''
        Returns attentions interpolated to input size (ndarray)
        '''
        # Squeeze channel
        atts = [self.att0.att.squeeze(1), self.att1.att.squeeze(1), self.att2.att.squeeze(1), self.att3.att.squeeze(1)]

        zoomed_atts = []
        for att in atts:
            assert att.shape[0] == 1, "Return atts only works when the previous prediction was of batch size 1"
            ishape = np.array(self.input_shape)
            ashape = np.array(att.shape[1:])
            assert len(ishape) == len(ashape)
            zoom_factors = (ishape/ashape).tolist()
            zoomed_atts.append(zoom(att.detach().cpu().numpy()[0], zoom_factors))

        return zoomed_atts


class UNetDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        '''
        Expected args:
        n_classes, init_channel, norm, dim
        '''
        super().__init__()
        n_classes = kwargs["n_classes"]
        init_channel = kwargs["init_channel"]
        norm = kwargs["norm"]
        dim = kwargs["dim"]
        self.up1 = Up(16*init_channel, 4*init_channel, norm, dim=dim)
        self.up2 = Up(8*init_channel, 2*init_channel,  norm, dim=dim)
        self.up3 = Up(4*init_channel, init_channel, norm, dim=dim)
        self.up4 = Up(2*init_channel, init_channel, norm, dim=dim)
        self.outc = getattr(nn, f"Conv{dim}")(init_channel, n_classes, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, y, out_1, out_2, out_3, out_4):
        y = self.up1(y, out_4)
        y = self.up2(y, out_3)
        y = self.up3(y, out_2)
        y = self.up4(y, out_1)
        y = self.outc(y)

        return y

class UNet(nn.Module):
    '''
    Main model class, final version
    Fixing design choices. 
    For old tunable unet check git history
    Removed residual, small, sigmoid and softmax applications
    '''
    def __init__(self, n_channels, n_classes, norm, dim, init_channel):
        super(UNet, self).__init__()

        self.enc = UNetEncoder(n_channels=n_channels, init_channel=init_channel, norm=norm, dim=dim)
        self.dec = UNetDecoder(n_classes=n_classes, init_channel=init_channel, norm=norm, dim=dim)

        print(f"UNet in channels: {n_channels}"
              f" batch_norm: {norm} dim: {dim}"
              f" out_channels {n_classes} ")

    def forward(self, x):
        return self.dec(*self.enc(x))
        