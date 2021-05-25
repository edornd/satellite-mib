import torch
import torch.nn as nn
from typing import Type

from cvmodels.segmentation.backbones import Backbone
from cvmodels.segmentation.deeplab import ASPPModule, ASPPVariants


class DecoderV3Partial(nn.Sequential):
    """Decoder for DeepLabV3, consisting of a double convolution and a direct 16X upsampling.
    This is clearly not the best for performance, but, if memory is a problem, this can save a little space.
    """

    def __init__(self,
                 dropout: float = 0.1,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            batch_norm(256),
            nn.ReLU(inplace=True), nn.Dropout(p=dropout))


class DecoderV3Plus(nn.Module):
    """DeepLabV3+ decoder branch, with a skip branch embedding low level
    features (higher resolution) into the highly dimensional output. This typically
    produces much better results than a naive 16x upsampling.
    Original paper: https://arxiv.org/abs/1802.02611
    """

    def __init__(self,
                 low_level_channels: int,
                 output_stride: int = 16,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Returns a new Decoder for DeepLabV3+.
        The upsampling is divided into two parts: a fixed 4x from 128 to 512, and a 2x or 4x
        from 32 or 64 (when input=512x512) to 128, depending on the output stride.

        :param low_level_channels: how many channels on the lo-level skip branch
        :type low_level_channels: int
        :param output_stride: downscaling factor of the backbone, defaults to 16
        :type output_stride: int, optional
        :param output_channels: how many outputs, defaults to 1
        :type output_channels: int, optional
        :param batch_norm: batch normalization module, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super().__init__()
        low_up_factor = 4
        high_up_factor = output_stride / low_up_factor
        self.low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            batch_norm(48),
            nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=high_up_factor, mode="bilinear", align_corners=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            batch_norm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            batch_norm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.low_level(skip)
        x = self.upsample(x)
        return self.output(torch.cat((skip, x), dim=1))




class DeeplabV3Head(nn.Module):
    def __init__(self,
                 backbone: Backbone,
                 in_dimension: int = 512,
                 aspp_variant: ASPPVariants = ASPPVariants.OS08,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        output_dims = in_dimension // backbone.scaling_factor()
        features_high, _ = backbone.output_features()
        self.aspp = ASPPModule(in_tensor=(features_high, output_dims, output_dims),
                               variant=aspp_variant,
                               batch_norm=batch_norm)
        self.decoder = DecoderV3Partial(batch_norm=batch_norm)
        self.apply(self.weight_reset)

    def forward(self, x, *args, **kwargs):
        x = self.aspp(x)
        return self.decoder(x)

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()


class DeeplabV3PlusHead(DeeplabV3Head):
    def __init__(self,
                 backbone: Backbone,
                 in_dimension: int = 512,
                 output_stride: int = 16,
                 aspp_variant: ASPPVariants = ASPPVariants.OS08,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__(backbone, in_dimension, aspp_variant, batch_norm)
        output_dims = in_dimension // backbone.scaling_factor()
        features_high, features_low = backbone.output_features()
        self.aspp = ASPPModule(in_tensor=(features_high, output_dims, output_dims),
                               variant=aspp_variant,
                               batch_norm=batch_norm)
        self.decoder = DecoderV3Plus(low_level_channels=features_low,
                                     output_stride=output_stride,
                                     batch_norm=batch_norm)
        self.apply(self.weight_reset)

    def forward(self, x, skip):
        x = self.aspp(x)
        return self.decoder(x, skip)
