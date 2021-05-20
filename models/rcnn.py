import traceback

import torch
from torch import nn
from torch.nn import functional as func
import torchvision
from .general import ResNetBlock


def test_seg_rcnn(model, device, logger):
    try:
        with torch.no_grad():
            dummy = torch.rand(2, 3, 800, 800).to(device)
            dummy_output_1, dummy_output_2 = model(dummy)
            dummy = dummy.to('cpu')
            dummy_output_1 = dummy_output_1.to('cpu')
            dummy_output_2 = dummy_output_2.to('cpu')
            del dummy, dummy_output_1, dummy_output_2
        logger.info('passed')
        return True
    except Exception as err:
        logger.critical(str(err))
        var = traceback.format_exc()
        logger.critical(var)
        return False


def seg_rcnn(config):
    decoder_channels = config.MODEL.PARAMS.DECODER_CHANNELS
    return SegRCNN(decoder_channels)


class SegRCNN(nn.Module):
    def __init__(self, decoder_channels=None, encoder_type='mobilenet'):
        super(SegRCNN, self).__init__()
        self._encoder_forward = {'mobilenet': self._mobilenet_forward,
                                 'resnet': self._resnet_forward}
        if encoder_type not in self._encoder_forward.keys():
            raise AssertionError(f'encoder_type should be one of {self._encoder_forward.keys()}')
        self._encoder_type = encoder_type
        if encoder_type == 'mobilenet':
            self.encoder = torchvision.models.mobilenet_v3_large(pretrained=True).features
            self._encoder_channels = [self.encoder[-1].out_channels, self.encoder[6].out_channels,
                                      self.encoder[3].out_channels, self.encoder[1].out_channels]
        else:
            self.encoder = torchvision.models.resnet101(pretrained=True)
            self.encoder.avgpool = torch.nn.Identity()
            self.encoder.fc = torch.nn.Identity()
            self._encoder_channels = [list(getattr(self.encoder, f'layer{i}').state_dict().values())[-2].shape[0]
                                      for i in range(4, 0, -1)]

        if decoder_channels is not None:
            self._decoder_channels = decoder_channels
            self._decoder_channels.insert(0, self._encoder_channels[0])
        else:
            self._decoder_channels = [self._encoder_channels[0], 512, 256, 128, 64, 32]

        self.semantic_features = list()
        for idx, decoder_channels in enumerate(self._decoder_channels[1:]):
            if (idx >= 1) and (idx <= 3):
                decoder_channels = decoder_channels - self._encoder_channels[idx]
            self.semantic_features.append(ResNetBlock(in_chn=self._decoder_channels[idx],
                                                      out_chn=decoder_channels,
                                                      stride=1,
                                                      expand=2))
        self.semantic_features = nn.Sequential(*self.semantic_features)

        self.last_block = list()
        self.last_block.append(ResNetBlock(in_chn=self._decoder_channels[-1],
                                           out_chn=self._decoder_channels[-1],
                                           stride=1,
                                           expand=2))
        self.last_block = nn.Sequential(*self.last_block)

    def _mobilenet_forward(self, x):
        x = self.encoder[0](x)
        skip_f_1 = self.encoder[1](x)
        x = self.encoder[2](skip_f_1)
        skip_f_3 = self.encoder[3](x)
        x = self.encoder[4](skip_f_3)
        x = self.encoder[5](x)
        skip_f_6 = self.encoder[6](x)
        x = self.encoder[7](skip_f_6)
        for block in self.encoder[8:]:
            x = block(x)
        return skip_f_1, skip_f_3, skip_f_6, x

    def _resnet_forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        skip_f_1 = self.encoder.layer1(x)
        skip_f_2 = self.encoder.layer2(skip_f_1)
        skip_f_3 = self.encoder.layer3(skip_f_2)
        x = self.encoder.layer4(skip_f_3)
        return skip_f_1, skip_f_2, skip_f_3, x

    def forward(self, x):
        skip_1, skip_2, skip_3, x = self._encoder_forward[self._encoder_type](x)

        for i in range(1, 6):
            x = self.semantic_features[i - 1](x)
            x = func.interpolate(x, scale_factor=2, mode='nearest')
            if i == 2:
                x = torch.cat((x, skip_3), 1)
            elif i == 3:
                x = torch.cat((x, skip_2), 1)
            elif i == 4:
                x = torch.cat((x, skip_1), 1)

        x = self.last_block(x)

        return x
