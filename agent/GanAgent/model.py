import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, channels_noise, features_g=64):
        super(Generator, self).__init__()
        # CONV TRANSPOSE PART
        self.generator = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._conv_transpose_block(channels_noise, features_g * 16, 4, 1, 0),
            # N x 1024 x 4 x 4
            self._conv_transpose_block(features_g * 16, features_g * 8, 4, 2, 1),
            # N x 512 x 8 x 8
            self._conv_transpose_block(features_g * 8, features_g * 4, 4, 2, 1),
            # N x 256 x 16 x 16
            self._conv_transpose_block(features_g * 4, 4, 4, 2, 1),
            # N x 4 x 32 x 32
            self._conv_block(4, 4, 3, 2, 1),
            # N x 4 x 16 x 16
            nn.MaxPool2d(kernel_size=2),
            # N x 4 x 8 x 8
            self._conv_block(4, 4, 3, 2, 1),
            # N x 4 x 4 x 4
            nn.MaxPool2d(kernel_size=2),
            # N x 4 x 2 x 2
            nn.Conv2d(4, 4, 3, 2, 1),
        )

        self.activation = nn.Tanh()

    def _conv_transpose_block(
        self, in_channels, out_channels, kernel_size, stride, padding
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)
        out = self.generator(x)
        return (self.activation(out) * 2).reshape(-1, out.size(1))


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
