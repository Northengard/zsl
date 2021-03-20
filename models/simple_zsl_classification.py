import torch
from torch import nn

from models.general import ConvBNReLUPool


def base_model(config):
    embedding_model = SimpleEmbeddingModel(input_shape=config.DATASET.PARAMS.IMAGE_SIZE,
                                           n_blocks=config.MODEL.PARAMS.N_BLOCKS,
                                           start_out_channels=config.MODEL.PARAMS.START_OUT_CHN,
                                           chn_multiplyer=config.MODEL.PARAMS.CHN_SCALE_COEF,
                                           vector_size=config.MODEL.PARAMS.VECTOR_SIZE)
    model = SiameseNet(embedding_net=embedding_model)
    return model


class SimpleEmbeddingModel(nn.Module):
    def __init__(self, input_shape, n_blocks=4, start_out_channels=2, chn_multiplyer=2, vector_size=200):
        super(SimpleEmbeddingModel, self).__init__()
        layers = [ConvBNReLUPool(1, start_out_channels)]
        current_chn_num = start_out_channels
        for _ in range(1, n_blocks):
            layers.append(ConvBNReLUPool(current_chn_num, current_chn_num * chn_multiplyer))
            current_chn_num *= chn_multiplyer
        self.encoder = nn.Sequential(*layers)
        reduce = 2**n_blocks
        input_shape = (input_shape[0] // reduce, input_shape[1] // reduce)
        num_features = input_shape[0] * input_shape[1] * current_chn_num
        self.feature_vectorizer = nn.Sequential(nn.Linear(num_features, vector_size, bias=False),
                                                nn.Linear(vector_size, vector_size, bias=True))

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.feature_vectorizer(x)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def get_embedding(self, x):
        return self.embedding_net(x)

    def forward(self, x):
        """
        SiameseNet forward
        :param x: x is supposed to be Nx2xCxHxW
        :return:
        """
        features = list()
        for idx in range(2):
            x_ = x[:, idx]
            x_ = self.embedding_net(x_)
            features.append(x_)
        return torch.stack(features, 1)
