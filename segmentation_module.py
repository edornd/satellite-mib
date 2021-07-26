from functools import partial, reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync

from apex.parallel import convert_syncbn_model
from modules.deeplab import DeepLabV3, DeepLabV3Plus
from modules.unet import UNet
from utils.utils import expand_input

norm_layers = {
    "iabn_sync": partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01),
    "iabn": partial(InPlaceABN, activation="leaky_relu", activation_param=.01),
    "abn": partial(ABN, activation="leaky_relu", activation_param=.01),
}

segment_heads = {"deeplabv3": DeepLabV3, "deeplabv3p": DeepLabV3Plus, "unet": UNet}


def add_ir_input(state_dict: Dict[str, Any], layer_name: str = "mod1.conv1.weight"):
    input_conv = state_dict[layer_name]
    r_kernels = input_conv[:, 0, :, :].unsqueeze(1)
    state_dict[layer_name] = torch.cat((input_conv, r_kernels), dim=1)
    return state_dict


def load_pretrained(pretrained_path: Path, model: nn.Module, input_channels: int = None) -> nn.Module:
    # load pretrained weights, if present LEGACY CODE
    # if opts.pretrained:
    #     pretrained_path = Path("pretrained") / f"{opts.backbone}_{opts.norm_act}.pth.tar"
    #     body = load_pretrained(pretrained_path, model=body)
    assert pretrained_path.exists(), f"Specified path '{pretrained_path}' does not exist"
    pre_dict = torch.load(pretrained_path, map_location='cpu')
    del pre_dict['state_dict']['classifier.fc.weight']
    del pre_dict['state_dict']['classifier.fc.bias']
    if input_channels and input_channels == 4:
        pre_dict["state_dict"] = add_ir_input(pre_dict["state_dict"])
    model.load_state_dict(pre_dict['state_dict'])
    del pre_dict    # free memory
    return model


def filter_arguments(backbone: str, **kwargs: Dict[str, Any]) -> bool:
    custom_layers = ("norm_layer", "act_layer")
    if backbone.startswith("tresnet"):
        for key in custom_layers + ("output_stride",):
            kwargs.pop(key, None)
    elif not backbone.startswith("resnet"):
        for key in custom_layers:
            kwargs.pop(key, None)
    return kwargs


def get_feature_indices(head: str, backbone: str) -> Tuple[int, ...]:
    # tresnet doesn't support out indices, but it has 4 of them
    if backbone.startswith("tresnet"):
        return None
    if head == "deeplabv3":
        return (4,)
    elif head == "deeplabv3p":
        return (1, 4)
    elif head == "unet":
        return tuple([i for i in range(5)])
    return None


def make_model(opts, classes=None):
    norm_layer = norm_layers.get(opts.norm_act, nn.BatchNorm2d)
    act_layer = partial(nn.ReLU, inplace=True) if norm_layer == "std" else nn.Identity
    additional_args = dict(norm_layer=norm_layer, act_layer=act_layer)
    additional_args = filter_arguments(opts.backbone)

    body = timm.create_model(opts.backbone,
                             pretrained=opts.pretrained,
                             checkpoint_path=None,
                             features_only=True,
                             out_indices=get_feature_indices(opts.head, opts.backbone),
                             **additional_args)
    # get the first set of 7x7 layers in the weights with shape [64, 3, 7, 7]
    if opts.input_channels > 3:
        print(f"Expanding input to handle {opts.input_channels} channels")
        body = expand_input(body, input_layer=None, copy_channel=0)

    head_class = segment_heads.get(opts.head)
    assert head_class is not None, f"Unknown decoder head: '{opts.head}'"
    head = head_class(feature_info=body.feature_info,
                      input_size=opts.crop_size,
                      output_stride=opts.output_stride,
                      num_classes=None,
                      act_layer=act_layer,
                      norm_layer=norm_layer)

    model = IncrementalSegmentationModule(body, head, head.output_channels(), classes=classes)
    model = convert_syncbn_model(model)
    return model


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body: nn.Module, head: nn.Module, head_channels: int, classes: List[int]):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        if not isinstance(classes, list):
            raise ValueError("Classes must be a list where every index refers to the n. of classes for that task")
        self.classifiers = nn.ModuleList([nn.Conv2d(head_channels, c, 1) for c in classes])
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):
        x_b = self.body(x)
        x_pl = self.head(*x_b)
        out = []
        for mod in self.classifiers:
            out.append(mod(x_pl))
        x_o = torch.cat(out, dim=1)
        if ret_intermediate:
            return x_o, x_b, x_pl
        return x_o

    def init_new_classifier(self, device: str) -> None:
        cls = self.classifiers[-1]
        imprinting_w = self.classifiers[0].weight[0]
        bkg_bias = self.classifiers[0].bias[0]
        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)
        new_bias = (bkg_bias - bias_diff)
        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)
        self.classifiers[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x: torch.Tensor, ret_intermediate: bool = False) -> Tuple[torch.Tensor, dict]:
        out_size = x.shape[-2:]
        out = self._network(x, ret_intermediate)
        sem_logits = out[0] if ret_intermediate else out
        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
        if ret_intermediate:
            return sem_logits, {"body": out[1], "pre_logits": out[2]}
        return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
