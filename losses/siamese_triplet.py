import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from .arcface_loss import ArcFaceLoss


class CAFLoss(nn.Module):
    def __init__(self, config):
        super(CAFLoss, self).__init__()
        self.afl = ArcFaceLoss(config)
        self.contrastive = ContrastiveLoss(config)
        self.afl_coef = config.LOSS.PARAMS.AFL_COEF
        self.contrastive_coef = config.LOSS.PARAMS.CONTRAST_COEF
        self.embedding_size = config.MODEL.PARAMS.VECTOR_SIZE

    def forward(self, embeddings, labels):
        output = torch.reshape(embeddings, shape=(-1, self.embedding_size))
        target = torch.reshape(labels, shape=(-1,))
        afl_loss = self.afl(output, target)
        contrastive_loss = self.contrastive(output, target)
        return self.afl_coef * afl_loss + self.contrastive_coef * contrastive_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.margin = config.LOSS.PARAMS.MARGIN
        self.embedding_size = config.MODEL.PARAMS.VECTOR_SIZE
        self.temperature = 2
        self.scale = 10
        self.eps = 1e-9
        self.criterion = losses.ContrastiveLoss(pos_margin=0,
                                                neg_margin=config.LOSS.PARAMS.MARGIN)

    def forward(self, embeddings, labels):
        output = torch.reshape(embeddings, shape=(-1, self.embedding_size))
        target = torch.reshape(labels, shape=(-1,))
        return self.criterion(output, target)
