from .siamese_triplet import ContrastiveLoss
from .discriminative import DiscriminativeLoss
from .arcface_loss import SegArcFace, ArcFaceLoss

__all__ = ['ContrastiveLoss', 'DiscriminativeLoss', 'SegArcFace', 'ArcFaceLoss']
