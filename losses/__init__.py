from .siamese_triplet import ContrastiveLoss
from .discriminative import DiscriminativeLoss
from .arcface_loss import SegArcFace

__all__ = ['ContrastiveLoss', 'DiscriminativeLoss', 'SegArcFace']
