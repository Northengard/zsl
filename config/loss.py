from yacs.config import CfgNode

_CONTRUSTIVE_LOSS = CfgNode()
_CONTRUSTIVE_LOSS.MARGIN = 1.

_DISCRIMINATIVE = CfgNode()
_DISCRIMINATIVE.SCALE_VAR = 1.
_DISCRIMINATIVE.SCALE_DIST = 1.
_DISCRIMINATIVE.SCALE_REG = 0.5
_DISCRIMINATIVE.DELTA_V = 0.5
_DISCRIMINATIVE.DELTA_D = 3.

_SEGARCFACE = CfgNode()
_SEGARCFACE.NUM_CLS = 80


_CAF_LOSS = CfgNode()
_CAF_LOSS.MARGIN = 1.
_CAF_LOSS.NUM_CLS = 80
_CAF_LOSS.AFL_COEF = 0.5
_CAF_LOSS.CONTRAST_COEF = 0.5


LOSS_DEFAULTS = {'ContrastiveLoss': _CONTRUSTIVE_LOSS,
                 'DiscriminativeLoss': _DISCRIMINATIVE,
                 'SegArcFace': _SEGARCFACE,
                 'ArcFaceLoss': _SEGARCFACE,
                 'CAFLoss': _CAF_LOSS,
                 'LargeMarginSoftmaxLoss': _SEGARCFACE,
                 'MultiSimilarityLoss': _SEGARCFACE}
