import logging
import torch
import torch.nn as nn
import torch.nn.functional as functional

from functools import reduce

from cvmodels.segmentation.backbones import resnet as rn
from cvmodels.segmentation.backbones import xception as xc
from cvmodels.segmentation.deeplab import  DeepLabVariants
from modules.deeplab import DeeplabV3Head, DeeplabV3PlusHead


logger = logging.getLogger(__name__)
model_variations = {
    "resnet101": {
        16: DeepLabVariants.RESNET101_16,
        8: DeepLabVariants.RESNET101_08
    },
    "resnet50": {
        16: DeepLabVariants.RESNET50_16,
        8: DeepLabVariants.RESNET50_08
    },
    "xception": {
        16: DeepLabVariants.XCEPTION16_16,
        8: DeepLabVariants.XCEPTION16_08
    }
}


def make_model(opts, classes=None):
    if "abn" in opts.norm_act:
        logger.warning("In-place batch normalization not working!")
    norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex

    variant: DeepLabVariants = None
    pretrained = not opts.no_pretrained
    # ResNet backbones
    if opts.backbone and opts.backbone.startswith("resnet"):
        variant = model_variations.get(opts.backbone).get(opts.output_stride, 16)
        backbone_variant, output_strides, _ = variant.value
        body = rn.ResNetBackbone(in_channels=opts.input_channels,
                                 variant=backbone_variant,
                                 output_strides=output_strides,
                                 batch_norm=norm,
                                 pretrained=pretrained)
    # Xception backbones
    elif opts.backbone and opts.backbone.startswith("xception"):
        variant = model_variations.get(opts.backbone).get(opts.output_stride, 16)
        backbone_variant, output_strides, _ = variant.value
        body = xc.XceptionBackbone(in_channels=opts.input_channels,
                                   output_strides=output_strides,
                                   variant=backbone_variant,
                                   batch_norm=norm,
                                   pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone '{opts.backbone}'")

    _, _, aspp_var = variant.value
    head_channels = 256
    if opts.head == "v3":
        head = DeeplabV3Head(backbone=body,
                            in_dimension=512,
                            aspp_variant=aspp_var,
                            batch_norm=norm)
    elif opts.head == "v3plus":
        head = DeeplabV3PlusHead(backbone=body,
                                 in_dimension=512,
                                 output_stride=opts.output_stride,
                                 aspp_variant=aspp_var,
                                 batch_norm=norm)
    else:
        raise ValueError(f"Unknown decoder head '{opts.head}'")

    if classes is not None:
        model = IncrementalSegmentationModule(body, head, head_channels, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"
        self.cls = nn.ModuleList(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):
        x_b = self.body(x)
        x_pl = self.head(*x_b)
        out = []
        for mod in self.cls:
            out.append(mod(x_pl))
        x_o = torch.cat(out, dim=1)
        if ret_intermediate:
            return x_o, x_b,  x_pl
        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]
        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)
        new_bias = (bkg_bias - bias_diff)
        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)
        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
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
