import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        # input size: (batch_size, 12, 60)
        # 12 = ohcl + noise
        # 60 = rows from ohlc
        # output size: (batch_size, 4, 60)

        self.conv_blocks = nn.Sequential(
            self._conv_block(input_channel, 128, 3, stride=2, padding=1),  # (128, 30)
            self._conv_block(128, 256, 3, stride=2, padding=1),  # (256, 15)
            self._conv_block(256, 256, 3, stride=1, padding=1),  # (256, 15)
        )

        self.t_conv_blocks = nn.Sequential(
            self._tconv_block(256, 128, 1, stride=1, padding=0),  # (128, 15)
            self._tconv_block(128, 64, 4, stride=2, padding=1),  # (64, 30)
            self._tconv_block(64, 12, 4, stride=2, padding=1),  # (12, 60)
        )
        self.last_conv = nn.ConvTranspose1d(12, 4, 3, stride=1, padding=1)  # (4, 60)

        self.last_activation = nn.Tanh()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = self.t_conv_blocks(out)
        out = self.last_conv(out)
        return self.last_activation(out) + 1

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )

    def _tconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )