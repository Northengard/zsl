import torch
from torch import nn
from pytorch_metric_learning import losses


# loss for mmdet
class BaseLoss(nn.Module):
    def __init__(self, config):
        super(BaseLoss, self).__init__()
        self.num_classes = config.LOSS.PARAMS.NUM_CLS
        self.embedding_size = config.MODEL.PARAMS.VECTOR_SIZE
        self.criterion = getattr(losses, config.LOSS.NAME)(num_classes=self.num_classes,
                                                           embedding_size=self.embedding_size)
        self.size_average = config.LOSS.SIZE_AVERAGE

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        mask = labels > 0
        real_labels = labels[mask] - 1
        loss = self.criterion(preds[mask], real_labels.to(torch.long))

        if self.size_average:
            loss /= batch_size
        return loss


class LargeMarginSoftmaxLoss(BaseLoss):
    def __init__(self, config):
        super(LargeMarginSoftmaxLoss, self).__init__(config)


class MultiSimilarityLoss(nn.Module):
    def __init__(self, config):
        super(MultiSimilarityLoss, self).__init__()
        self.num_classes = config.LOSS.PARAMS.NUM_CLS + 1
        self.embedding_size = config.MODEL.PARAMS.VECTOR_SIZE
        self.criterion = losses.MultiSimilarityLoss()
        self.size_average = config.LOSS.SIZE_AVERAGE

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        mask = labels > 0
        real_labels = labels[mask] - 1
        loss = self.criterion(preds[mask], real_labels.to(torch.long))

        if self.size_average:
            loss /= batch_size
        return loss
