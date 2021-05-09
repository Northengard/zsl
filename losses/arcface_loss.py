import torch
from torch import nn
from mmdet.models.builder import LOSSES
from pytorch_metric_learning import losses


class SegArcFace(nn.Module):
    def __init__(self, config):
        super(SegArcFace, self).__init__()
        self.criterion = losses.ArcFaceLoss(num_classes=config.LOSS.PARAMS.NUM_CLS,
                                            embedding_size=config.MODEL.PARAMS.VECTOR_SIZE)
        self.size_average = config.LOSS.SIZE_AVERAGE

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        mask = labels > 0
        loss = self.criterion(preds[mask], labels[labels > 0])

        if self.size_average:
            loss /= batch_size
        return loss


# loss for mmdet
@LOSSES.register_module()
class ArcFaceLoss(torch.nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.criterion = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        loss = self.criterion(preds, labels)

        if self.size_average:
            loss /= batch_size
        return loss
