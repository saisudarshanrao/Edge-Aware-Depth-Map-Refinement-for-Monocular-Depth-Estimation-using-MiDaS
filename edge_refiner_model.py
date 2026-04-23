from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(8, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EdgeGuidedDepthRefiner(nn.Module):
    """
    Input:  RGB(3) + raw_depth(1) + edge_map(1) = 5 channels
    Output: residual correction map (1 channel)
    Final depth = clamp(raw_depth + residual, 0, 1)
    """

    def __init__(self, in_channels: int = 5, base_channels: int = 32, residual_scale: float = 0.25):
        super().__init__()
        self.residual_scale = residual_scale

        self.stem = ConvBlock(in_channels, base_channels)          # 32
        self.down1 = DownBlock(base_channels, base_channels * 2)   # 64
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)  # 256

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)

        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

        # Start from near-identity behavior
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        b = self.down3(s3)
        b = self.bottleneck(b)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        residual = torch.tanh(self.out_conv(d1)) * self.residual_scale
        return residual