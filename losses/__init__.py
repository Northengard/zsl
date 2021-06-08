from .siamese_triplet import ContrastiveLoss, CAFLoss
from .discriminative import DiscriminativeLoss
from .arcface_loss import SegArcFace, ArcFaceLoss
from .top_down_metric_learning_losses import MultiSimilarityLoss, LargeMarginSoftmaxLoss

__all__ = ['ContrastiveLoss', 'DiscriminativeLoss',
           'SegArcFace', 'ArcFaceLoss', 'CAFLoss',
           'MultiSimilarityLoss', 'LargeMarginSoftmaxLoss']
