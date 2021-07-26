from typing import Any, Dict, Tuple
from utils.utils import expand_input
from modules.unet import UNet
from modules.deeplab import DeepLabV3, DeepLabV3Plus
import timm
import torch
from torch import nn

segment_heads = {"deeplabv3": DeepLabV3, "deeplabv3p": DeepLabV3Plus, "unet": UNet}


def add_ir_input(state_dict: Dict[str, Any], layer_name: str = "mod1.conv1.weight"):
    input_conv = state_dict[layer_name]
    r_kernels = input_conv[:, 0, :, :].unsqueeze(1)
    state_dict[layer_name] = torch.cat((input_conv, r_kernels), dim=1)
    return state_dict


def filter_arguments(backbone: str, **kwargs: Dict[str, Any]) -> bool:
    custom_layers = ("norm_layer", "act_layer")
    if backbone.startswith("tresnet"):
        for key in custom_layers + ("output_stride",):
            kwargs.pop(key, None)
    elif not backbone.startswith("resnet"):
        for key in custom_layers:
            kwargs.pop(key, None)
    return kwargs


def get_feature_indices(head: str) -> Tuple[int, ...]:
    if head == "deeplabv3":
        return (4,)
    elif head == "deeplabv3p":
        return (1, 4)
    elif head == "unet":
        return tuple([i for i in range(5)])
    return None


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body: nn.Module, head: nn.Module):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x_b = self.body(x)
        return self.head(*x_b)


input_channels = 3
additional_args = dict()
body = timm.create_model("resnet50",
                         pretrained=True,
                         checkpoint_path=None,
                         features_only=True,
                         out_indices=get_feature_indices("unet"),
                         **additional_args)
# get the first set of 7x7 layers in the weights with shape [64, 3, 7, 7]
if input_channels > 3:
    print(f"Expanding input to handle {input_channels} channels")
    body = expand_input(body, input_layer=None, copy_channel=0)

head_class = segment_heads.get("unet")
head = head_class(feature_info=body.feature_info, input_size=512, output_stride=16, num_classes=None)
model = IncrementalSegmentationModule(body, head)

out = model(torch.rand(4, 3, 512, 512))
print(out.shape)
