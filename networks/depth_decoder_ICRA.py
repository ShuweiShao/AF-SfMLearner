from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from torch.nn.parameter import Parameter


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales = range(4) , num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict() # 有序字典
        for i in range(4, -1, -1):
            
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.convs[("attention", 0)] = ChannelSpatialSELayer(self.num_ch_enc[-1], reduction_ratio=16)
        # self.convs[("attention", 0)] = ChannelSELayer(self.num_ch_enc[-1], reduction_ratio=16)
        # self.convs[("attention", 0)] = SPPSELayer(self.num_ch_enc[-1], reduction=16)
        # self.convs[("attention", 0)] = SpatialSELayer(self.num_ch_enc[-1])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        x = self.convs[("attention", 0)](x)
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=16):

        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.elu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        
        return output_tensor


class SPPSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SPPSELayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel*21, channel*21 // reduction, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(channel*21 // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)


class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        
        self.conv0 = nn.Conv2d(num_channels, num_channels // 16, 1)
        self.conv1 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=1)
        self.conv2 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=2)
        self.conv3 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=3)
        self.conv4 = nn.Conv2d(num_channels // 16, num_channels // 16, 3, dilation=4)
        self.conv5 = nn.Conv2d((num_channels // 16), 1, 1)
        self.elu = nn.ELU(inplace=True)
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(2)
        self.pad3 = nn.ReflectionPad2d(3)
        self.pad4 = nn.ReflectionPad2d(4)
        self.sigmoid = nn.Sigmoid()
        self.p1 = Parameter(torch.ones(1))
        self.p2 = Parameter(torch.zeros(1))
        self.p3 = Parameter(torch.zeros(1))
        self.p4 = Parameter(torch.zeros(1))

        
    def forward(self, input_tensor):

        batch_size, channel, a, b = input_tensor.size()

        out0 = self.conv0(input_tensor)

        out1 = self.pad1(out0)
        out1 = self.conv1(out1)
        out1 = self.elu(out1)
        att1 = self.conv5(out1)
        att1 = self.sigmoid(att1)

        out2 = torch.add(out0, out1)
        out2 = self.pad2(out2)
        out2 = self.conv2(out2)
        out2 = self.elu(out2)
        att2 = self.conv5(out2)
        att2 = self.sigmoid(att2)      

        out3 = torch.add(out0, out2)
        out3 = self.pad3(out3)
        out3 = self.conv3(out3)
        out3 = self.elu(out3)
        att3 = self.conv5(out3)
        att3 = self.sigmoid(att3)      

        out4 = torch.add(out0, out3)
        out4 = self.pad3(out4)
        out4 = self.conv3(out4)
        out4 = self.elu(out4)
        att4 = self.conv5(out4)
        att4 = self.sigmoid(att4)

        att1 = att1.view(batch_size, 1, a, b)
        att2 = att2.view(batch_size, 1, a, b)
        att3 = att3.view(batch_size, 1, a, b)
        att4 = att4.view(batch_size, 1, a, b)

        out1 = torch.mul(input_tensor, att1)
        out2 = torch.mul(input_tensor, att2)
        out3 = torch.mul(input_tensor, att3)
        out4 = torch.mul(input_tensor, att4)

        output_tensor = self.elu((self.p1 * out1 + self.p2 * out2 + self.p3 * out3 + self.p4 * out4))
        # output_tensor = self.elu(out1 + out2 + out3 + out4)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=16):

        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = SPPSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        self.conv = nn.Conv2d(2 * num_channels, num_channels, 1)
        self.elu = nn.ELU()

    def forward(self, input_tensor):

        output_tensor = torch.cat((self.cSE(input_tensor), self.sSE(input_tensor)), dim=1)
        output_tensor = self.conv(output_tensor)
        output_tensor = torch.add(input_tensor, output_tensor)
        output_tensor = self.elu(output_tensor)

        return output_tensor


