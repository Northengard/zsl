from yacs.config import CfgNode

_CONSTRUCT_LOSS = CfgNode()
_CONSTRUCT_LOSS.MARGIN = 1.

_DISCRIMINATIVE = CfgNode()
_DISCRIMINATIVE.SCALE_VAR = 1.
_DISCRIMINATIVE.SCALE_DIST = 1.
_DISCRIMINATIVE.SCALE_REG = 0.5
_DISCRIMINATIVE.DELTA_V = 0.5
_DISCRIMINATIVE.DELTA_D = 3.

_SEGARCFACE = CfgNode()
_SEGARCFACE.NUM_CLS = 80


LOSS_DEFAULTS = {'ContrastiveLoss': _CONSTRUCT_LOSS,
                 'DiscriminativeLoss': _DISCRIMINATIVE,
                 'SegArcFace': _SEGARCFACE,
                 'ArcFaceLoss': _SEGARCFACE}
