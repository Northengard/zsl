import torch
from torch import nn
from mmdet.models.builder import LOSSES
from pytorch_metric_learning import losses


class SegArcFace(nn.Module):
    def __init__(self, config):
        super(SegArcFace, self).__init__()
        self.criterion = losses.ArcFaceLoss(num_classes=config.LOSS.PARAMS.NUM_CLS + 1,
                                            embedding_size=config.MODEL.PARAMS.VECTOR_SIZE)
        self.size_average = config.LOSS.SIZE_AVERAGE

    def forward(self, preds, labels):
        semantic_emb, instance_emb = preds
        semantic_gt = labels[:, 0]
        instance_gt = labels[:, 1]
        batch_size = semantic_emb.shape[0]

        mask = semantic_gt > 0
        loss = self.criterion(preds[mask], semantic_gt[mask])

        mask = instance_gt > 0
        loss += self.criterion(instance_emb[mask], instance_emb[mask])

        if self.size_average:
            loss /= batch_size
        return loss


# loss for mmdet
class ArcFaceLoss(torch.nn.Module):
    def __init__(self, config):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = config.LOSS.PARAMS.NUM_CLS + 1
        self.embedding_size = config.MODEL.PARAMS.VECTOR_SIZE
        self.criterion = losses.ArcFaceLoss(num_classes=self.num_classes, embedding_size=self.embedding_size)
        self.size_average = config.LOSS.SIZE_AVERAGE

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        loss = self.criterion(preds, labels.to(torch.long))

        if self.size_average:
            loss /= batch_size
        return loss
