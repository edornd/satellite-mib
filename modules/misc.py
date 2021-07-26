import torch
import torch.nn as nn


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.SyncBatchNorm, nn.BatchNorm2d)):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
