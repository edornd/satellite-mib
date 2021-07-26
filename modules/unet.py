from functools import partial
from typing import Iterable, List, Type

import torch
from timm.models.features import FeatureInfo
from torch import nn

from modules import HeadlessSegmenter


class UNetDecodeBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 bilinear: bool = True,
                 act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        up_channels = in_channels // 2
        self.upsample = self._upsampling(in_channels, up_channels, factor=scale_factor, bilinear=bilinear)
        self.conv = self._upconv(up_channels + skip_channels, out_channels, act_layer=act_layer, norm_layer=norm_layer)
        self.adapter = nn.Conv2d(up_channels, out_channels, kernel_size=1, bias=True)

    def _upsampling(self, in_channels: int, out_channels: int, factor: int, bilinear: bool = True):
        if bilinear:
            return nn.Sequential(nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=factor, stride=factor)

    def _upconv(self,
                in_channels: int,
                out_channels: int,
                act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
                norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> nn.Sequential:
        # yapf: disable
        # mid_channels = (in_channels + out_channels) // 2
        mid_channels = out_channels
        return nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(mid_channels),
                             act_layer(),
                             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                             norm_layer(out_channels),
                             act_layer())
        # yapf: enable

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        combined = torch.cat((x, skip), dim=1)
        return self.adapter(x) + self.conv(combined)


class UNetHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, scale_factor: int = 2, dropout_prob: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        if num_classes is not None:
            self.out = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        else:
            self.out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.upsample(x)
        return self.out(x)


class DecoderUNet(nn.Module):

    def __init__(self,
                 feature_channels: List[int],
                 feature_reductions: List[int],
                 bilinear: bool = True,
                 output_channels: int = 1,
                 act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        # invert sequences to decode
        chs = feature_channels[::-1]
        reductions = feature_reductions[::-1] + [1]
        scales = [int(reductions[i] // reductions[i + 1]) for i in range(len(reductions) - 1)]

        self.up1 = UNetDecodeBlock(chs[0], chs[1], chs[0] // 2, scales[0], bilinear, act_layer, norm_layer)
        self.up2 = UNetDecodeBlock(chs[1], chs[2], chs[1] // 2, scales[1], bilinear, act_layer, norm_layer)
        self.up3 = UNetDecodeBlock(chs[2], chs[3], chs[2] // 2, scales[2], bilinear, act_layer, norm_layer)
        self.up4 = UNetDecodeBlock(chs[3], chs[4], chs[3] // 2, scales[3], bilinear, act_layer, norm_layer)
        self.out = UNetHead(chs[3] // 2, output_channels, scales[4])
        self.last_channels = chs[3] // 2

    def forward(self, *features: Iterable[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)


class UNet(HeadlessSegmenter):

    def __init__(self,
                 feature_info: FeatureInfo,
                 num_classes: int = 1,
                 act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 **kwargs) -> None:
        super().__init__()
        self.decoder = DecoderUNet(feature_channels=feature_info.channels(),
                                   feature_reductions=feature_info.reduction(),
                                   output_channels=num_classes,
                                   act_layer=act_layer,
                                   norm_layer=norm_layer)

    def forward(self, *features: Iterable[torch.Tensor]) -> torch.Tensor:
        return self.decoder(*features)

    def output_channels(self):
        return self.decoder.last_channels
