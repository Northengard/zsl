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
    return MobileNetSegRCNN(decoder_channels)


class MobileNetSegRCNN(nn.Module):
    def __init__(self, decoder_channels=None):
        super(MobileNetSegRCNN, self).__init__()
        self.encoder = torchvision.models.mobilenet_v3_large(pretrained=True).features
        self._encoder_channels = [self.encoder[-1].out_channels, self.encoder[6].out_channels,
                                  self.encoder[3].out_channels, self.encoder[1].out_channels]

        if decoder_channels is not None:
            self._decoder_channels = decoder_channels
            self._decoder_channels.insert(0, self.encoder[-1].out_channels)
        else:
            self._decoder_channels = [self.encoder[-1].out_channels, 512, 256, 128, 64, 32]

        self.semantic_features = list()
        for idx, decoder_channels in enumerate(self._decoder_channels[1:]):
            if (idx >= 1) and (idx <= 3):
                decoder_channels = decoder_channels - self._encoder_channels[idx]
            self.semantic_features.append(ResNetBlock(in_chn=self._decoder_channels[idx],
                                                      out_chn=decoder_channels,
                                                      stride=1,
                                                      expand=2))
        self.semantic_features = nn.Sequential(*self.semantic_features)

        self.instance_features = list()
        self.instance_features.append(ResNetBlock(in_chn=self._decoder_channels[-1],
                                                  out_chn=self._decoder_channels[-1],
                                                  stride=1,
                                                  expand=2))
        self.instance_features = nn.Sequential(*self.instance_features)

    def forward(self, x):
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

        for i in range(1, 6):
            x = self.semantic_features[i - 1](x)
            x = func.interpolate(x, scale_factor=2, mode='nearest')
            if i == 2:
                x = torch.cat((x, skip_f_6), 1)
            elif i == 3:
                x = torch.cat((x, skip_f_3), 1)
            elif i == 4:
                x = torch.cat((x, skip_f_1), 1)

        instance_embeddings = self.instance_features(x)

        return x, instance_embeddings
