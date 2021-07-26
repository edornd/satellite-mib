from torch import nn


class HeadlessSegmenter(nn.Module):

    def output_channels(self):
        NotImplementedError()
