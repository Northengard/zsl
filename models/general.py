from torch import nn


class ConvBNReLUPool(nn.Sequential):
    def __init__(self, in_chn, out_chn):
        super(ConvBNReLUPool, self).__init__(*[nn.Conv2d(in_channels=in_chn, out_channels=out_chn,
                                                         kernel_size=3, stride=1, padding=1),
                                               nn.Conv2d(in_channels=out_chn, out_channels=out_chn,
                                                         kernel_size=3, stride=1, padding=1),
                                               nn.BatchNorm2d(out_chn),
                                               nn.ReLU6(),
                                               nn.MaxPool2d(kernel_size=2, stride=2)])


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_chn, out_chn, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__(*[nn.Conv2d(in_channels=in_chn, out_channels=out_chn,
                                                     kernel_size=kernel_size, stride=stride, padding=padding),
                                           nn.BatchNorm2d(out_chn),
                                           nn.ReLU6()])


class ResNetBlock(nn.Module):
    def __init__(self, in_chn, out_chn, expand=1, stride=1):
        super(ResNetBlock, self).__init__()
        self._expansion = expand
        self._stride = stride
        self._intra_channels = self._expansion * in_chn
        self._input_channels = in_chn
        self._output_channels = out_chn
        self.layers = nn.Sequential(*[ConvBNReLU(in_chn=self._input_channels,
                                                 out_chn=self._intra_channels,
                                                 kernel_size=1, stride=1, padding=0),
                                      ConvBNReLU(in_chn=self._intra_channels,
                                                 out_chn=self._input_channels,
                                                 kernel_size=3, stride=1, padding=1),
                                      ConvBNReLU(in_chn=self._input_channels,
                                                 out_chn=self._output_channels,
                                                 kernel_size=3, stride=stride, padding=1)])

    @property
    def expansion(self):
        return self._expansion

    @property
    def stride(self):
        return self._stride

    def forward(self, x):
        x = self.layers(x)
        return x
