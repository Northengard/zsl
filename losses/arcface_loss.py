import torch
from torch import nn
from pytorch_metric_learning import losses


class SegArcFace(nn.Module):
    def __init__(self, config):
        super(SegArcFace, self).__init__()
        self.criterion = losses.ArcFaceLoss(num_classes=config.LOSS.NUM_CLS,
                                            embedding_size=config.MODEL.PARAMS.VECTOR_SIZE)

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        loss = torch.zeros(1)
        for pred_id, pred in enumerate(preds):
            loss += self.criterion(pred.reshape(-1), labels.reshape(-1))

        loss /= batch_size
        return loss
