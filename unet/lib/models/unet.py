#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inchannels, outchannels, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels,
                               kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(outchannels, outchannels,
                               kernel_size=3, padding=padding)
        self.batchnorm = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.leaky_relu(x)
        # Adding dropout
        x = F.dropout(x, p=0.5)
        return x        # Adding dropout

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(inchannels, outchannels,
                                         kernel_size=2, stride=2)
        self.conv = ConvBlock(inchannels, outchannels)
        # Adding batchnorm
        self.batchnorm = nn.BatchNorm2d(outchannels)

    def forward(self, x, locality_info):
        x = self.upconv(x)
        #x = self.batchnorm(x)
        x = torch.cat([locality_info, x], 1) # adding in dim = 1 which is channels.
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        return x    

class Unet(nn.Module):
    def __init__(self, inchannels, outchannels, net_depth):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_channels = inchannels
        out_channels = 64
        for _ in range(net_depth):
            conv = ConvBlock(in_channels, out_channels)
            self.downblocks.append(conv)
            in_channels, out_channels = out_channels, 2 * out_channels

        self.middle_conv = ConvBlock(in_channels, out_channels)

        in_channels, out_channels = out_channels, int(out_channels / 2)
        for _ in range(net_depth):
            upconv = UpBlock(in_channels, out_channels)
            self.upblocks.append(upconv)
            in_channels, out_channels = out_channels, int(out_channels / 2)

        self.seg_layer = nn.Conv2d(
            2 * out_channels, outchannels, kernel_size=1)

    def forward(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())

        x = self.seg_layer(x)
        # in case it's one output class, squeeze the channels out
        x = x.squeeze(dim=1)

        return x
        
def test_model():
    x = torch.randn((4,3,128,128))
    model = Unet(inchannels=3, outchannels=1, net_depth=3)
    print(model)
    m = torch.nn.LogSoftmax(dim=0)
    preds = m(model(x))
    print(f'input shape: {x.shape}')
    print(f'output shape: {preds.shape}')

if __name__ == "__main__":
    test_model()