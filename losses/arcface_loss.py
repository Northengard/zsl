import torch
from torch import nn
from pytorch_metric_learning import losses
from top_down_metric_learning_losses import BaseLoss


class SegArcFace(nn.Module):
    def __init__(self, config):
        super(SegArcFace, self).__init__()
        self.emb_size = config.MODEL.PARAMS.VECTOR_SIZE
        self.criterion = losses.ArcFaceLoss(num_classes=config.LOSS.PARAMS.NUM_CLS,
                                            embedding_size=config.MODEL.PARAMS.VECTOR_SIZE)
        self.size_average = config.LOSS.SIZE_AVERAGE

    def forward(self, preds, labels):
        batch_size = labels.shape[0]
        mask = labels > 0
        real_labels = labels[mask] - 1
        loss = self.criterion(preds[mask].reshape(-1, self.emb_size), real_labels.reshape(-1).to(torch.long))

        if self.size_average:
            loss /= batch_size
        return loss


class ArcFaceLoss(BaseLoss):
    def __init__(self, config):
        super(ArcFaceLoss, self).__init__(config)
