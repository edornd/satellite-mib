from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as functional
from models.util import try_index
from timm.models.features import FeatureInfo

from modules import HeadlessSegmenter
from modules.misc import initialize_weights


class DeeplabV3Old(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = 256,
                 out_stride: int = 16,
                 norm_act: Type[nn.Module] = nn.BatchNorm2d,
                 pooling_size: int = None):
        super(DeeplabV3Old, self).__init__()
        self.pooling_size = pooling_size

        if out_stride == 16:
            dilations = [6, 12, 18]
        elif out_stride == 8:
            dilations = [12, 24, 32]

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.Conv2d(in_channels, mid_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]),
            nn.Conv2d(in_channels, mid_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]),
            nn.Conv2d(in_channels, mid_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2])
        ])
        self.map_bn = norm_act(mid_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(mid_channels)

        self.red_conv = nn.Conv2d(mid_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.reset_parameters(self.map_bn.activation, self.map_bn.activation_param)

    def reset_parameters(self, activation, slope):
        gain = nn.init.calculate_gain(activation, slope)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)    # if training is global avg pooling 1x1, else use larger pool size
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            # this is like Adaptive Average Pooling (1,1)
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0),
                                x.shape[2]), min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = ((pooling_size[1] - 1) // 2, (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else
                       (pooling_size[1] - 1) // 2 + 1, (pooling_size[0] - 1) // 2,
                       (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1)

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
        return pool


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module: this block is responsible for the multi-scale feature extraction,
    using multiple parallel convolutional blocks (conv, bn, relu) with different dilations.
    The four feature groups are then recombined into a single tensor together with an upscaled average pooling
    (that contrasts information loss), then again processed by a 1x1 convolution + dropout
    """

    def __init__(self,
                 in_size: int = 32,
                 in_channels: int = 2048,
                 output_stride: int = 16,
                 out_channels: int = 256,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        dil_factor = int(output_stride // 16)    # equals 1 or 2 if os = 8
        dils = tuple(v * dil_factor for v in (1, 6, 12, 18))
        self.aspp1 = self.aspp_block(in_channels, 256, 1, 0, dils[0], act_layer=act_layer, batch_norm=batch_norm)
        self.aspp2 = self.aspp_block(in_channels, 256, 3, dils[1], dils[1], act_layer=act_layer, batch_norm=batch_norm)
        self.aspp3 = self.aspp_block(in_channels, 256, 3, dils[2], dils[2], act_layer=act_layer, batch_norm=batch_norm)
        self.aspp4 = self.aspp_block(in_channels, 256, 3, dils[3], dils[3], act_layer=act_layer, batch_norm=batch_norm)
        # this is redoncolous, but it's described in the paper: bring it down to 1x1 tensor and upscale
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels,
                                                                             256,
                                                                             kernel_size=1,
                                                                             bias=False), batch_norm(256), act_layer(),
                                     nn.Upsample((in_size, in_size), mode="bilinear", align_corners=True))
        self.merge = self.aspp_block(256 * 5, out_channels, 1, 0, 1, act_layer=act_layer, batch_norm=batch_norm)
        self.dropout = nn.Dropout(p=0.5)

    def aspp_block(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                   act_layer: Type[nn.Module], batch_norm: Type[nn.Module]) -> nn.Sequential:
        module = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      bias=False), batch_norm(out_channels), act_layer())
        return module

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x1 = self.aspp1(batch)
        x2 = self.aspp2(batch)
        x3 = self.aspp3(batch)
        x4 = self.aspp4(batch)
        x5 = self.avgpool(batch)
        x5 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.merge(x5)
        return self.dropout(x)


class EfficientASPP(ASPPModule):
    """Variant of the standard ASPP block, using a bottleneck approach similar to the ResNet one.
    The aim of the bottleneck is to reduce the number of parameters, while increasing the performance.
    Implementation follows: https://doi.org/10.1007/s11263-019-01188-y
    """

    def __init__(self,
                 in_size: int = 32,
                 in_channels: int = 2048,
                 output_stride: int = 16,
                 out_channels: int = 256,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super(ASPPModule, self).__init__()
        dil_factor = int(output_stride // 16)    # equals 1 or 2 if os = 8
        dils = tuple(v * dil_factor for v in (1, 6, 12, 18))
        self.aspp1 = self.aspp_block(in_channels, 256, 1, 0, dils[0], act_layer=act_layer, batch_norm=batch_norm)
        self.aspp2 = self.bottleneck(in_channels, 256, 3, dils[1], dils[1], act_layer=act_layer, batch_norm=batch_norm)
        self.aspp3 = self.bottleneck(in_channels, 256, 3, dils[2], dils[2], act_layer=act_layer, batch_norm=batch_norm)
        self.aspp4 = self.bottleneck(in_channels, 256, 3, dils[3], dils[3], act_layer=act_layer, batch_norm=batch_norm)
        # this is redoncolous, but it's described in the paper: bring it down to 1x1 tensor and upscale
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels,
                                                                             256,
                                                                             kernel_size=1,
                                                                             bias=False), batch_norm(256), act_layer(),
                                     nn.Upsample((in_size, in_size), mode="bilinear", align_corners=True))
        self.merge = self.aspp_block(256 * 5, out_channels, kernel=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.dropout = nn.Dropout(p=0.5)

    def bottleneck(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                   act_layer: Type[nn.Module], batch_norm: Type[nn.Module]) -> nn.Sequential:
        mid_channels = out_channels // 4
        modules = list()
        modules.extend(list(self.aspp_block(in_channels, mid_channels, 1, 0, 1, act_layer, batch_norm)))
        modules.extend(
            list(self.aspp_block(mid_channels, mid_channels, kernel, padding, dilation, act_layer, batch_norm)))
        modules.extend(
            list(self.aspp_block(mid_channels, mid_channels, kernel, padding, dilation, act_layer, batch_norm)))
        modules.extend(list(self.aspp_block(mid_channels, out_channels, 1, 0, 1, act_layer, batch_norm)))
        return nn.Sequential(*modules)


class DecoderV3(nn.Sequential):
    """Decoder for DeepLabV3, consisting of a double convolution and a direct 16X upsampling.
    This is clearly not the best for performance, but, if memory is a problem, this can save a little space.
    """

    def __init__(self,
                 in_channels: int = 256,
                 output_stride: int = 16,
                 output_channels: int = 1,
                 dropout: float = 0.1,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 include_head: bool = False):
        modules = [
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            batch_norm(256),
            act_layer(),
            nn.Dropout(p=dropout)
        ]
        if include_head:
            modules.append(nn.Conv2d(256, output_channels, kernel_size=1))
            modules.append(nn.Upsample(scale_factor=output_stride, mode="bilinear", align_corners=True))
        super(DecoderV3, self).__init__(*modules)


class DecoderV3Plus(nn.Module):
    """DeepLabV3+ decoder branch, with a skip branch embedding low level
    features (higher resolution) into the highly dimensional output. This typically
    produces much better results than a naive 16x upsampling.
    Original paper: https://arxiv.org/abs/1802.02611
    """

    def __init__(self,
                 skip_channels: int,
                 aspp_channels: int = 256,
                 output_stride: int = 16,
                 output_channels: int = 1,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 include_head: bool = False):
        super().__init__()
        low_up_factor = 4
        high_up_factor = output_stride / low_up_factor
        self.low_level = nn.Sequential(nn.Conv2d(skip_channels, 48, 1, bias=False), batch_norm(48), act_layer())
        self.upsample = nn.Upsample(scale_factor=high_up_factor, mode="bilinear", align_corners=True)

        # Table 2, best performance with two 3x3 convs, yapf: disable
        modules = [
            nn.Conv2d(48 + aspp_channels, 256, 3, stride=1, padding=1, bias=False),
            batch_norm(256),
            act_layer(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            batch_norm(256),
            act_layer(),
            nn.Dropout(0.1)]
        if include_head:
            modules.extend([
                nn.Conv2d(256, output_channels, 1, stride=1),
                nn.Upsample(scale_factor=low_up_factor, mode="bilinear", align_corners=True),
                nn.Dropout(0.1)
            ])
        self.output = nn.Sequential(*modules)
        # yapf: enable

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.low_level(skip)
        x = self.upsample(x)
        return self.output(torch.cat((skip, x), dim=1))


class DeepLabV3(HeadlessSegmenter):

    def __init__(self,
                 feature_info: FeatureInfo,
                 input_size: int = 512,
                 output_stride: int = 16,
                 aspp_channels: int = 256,
                 num_classes: int = 1,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        assert output_stride in (8, 16), f"Invalid output stride: '{output_stride}'"
        # only one layer
        channels = feature_info.channels()[0]
        reduction = feature_info.reduction()[0]
        include_head = num_classes is not None and num_classes > 0
        self.aspp = ASPPModule(in_size=int(input_size / reduction),
                               in_channels=channels,
                               output_stride=output_stride,
                               out_channels=aspp_channels,
                               act_layer=act_layer,
                               batch_norm=norm_layer)
        self.decoder = DecoderV3(in_channels=aspp_channels,
                                 output_stride=output_stride,
                                 output_channels=num_classes,
                                 include_head=include_head,
                                 act_layer=act_layer,
                                 batch_norm=norm_layer)
        self.aspp.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(*x)
        return self.decoder(x)

    def output_channels(self):
        return 256


class DeepLabV3Plus(HeadlessSegmenter):

    def __init__(self,
                 feature_info: FeatureInfo,
                 input_size: int = 512,
                 output_stride: int = 16,
                 aspp_channels: int = 256,
                 num_classes: int = 1,
                 act_layer: Type[nn.Module] = nn.ReLU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d) -> None:
        assert output_stride in (8, 16), f"Invalid output stride: '{output_stride}'"
        super().__init__()
        low_channels, high_channels = feature_info.channels()
        _, high_reduction = feature_info.reduction()
        self.aspp = ASPPModule(in_size=int(input_size / high_reduction),
                               in_channels=high_channels,
                               output_stride=output_stride,
                               out_channels=aspp_channels,
                               act_layer=act_layer,
                               batch_norm=norm_layer)
        self.decoder = DecoderV3Plus(skip_channels=low_channels,
                                     aspp_channels=aspp_channels,
                                     output_stride=output_stride,
                                     output_channels=num_classes,
                                     act_layer=act_layer,
                                     batch_norm=norm_layer)
        self.aspp.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    def forward(self, skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        return self.decoder(x, skip)

    def output_channels(self):
        return 256
