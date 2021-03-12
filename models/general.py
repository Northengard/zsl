from torch import nn


class ConvBNReLUPool(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(ConvBNReLUPool, self).__init__()
        self.layers = nn.Sequential(*[nn.Conv2d(in_channels=in_chn, out_channels=out_chn,
                                                kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(in_channels=out_chn, out_channels=out_chn,
                                                kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_chn),
                                      nn.ReLU6(),
                                      nn.MaxPool2d(kernel_size=2, stride=2)])

    def forward(self, x):
        x = self.layers(x)
        return x
