from yacs.config import CfgNode

_CONSTRUCT_LOSS = CfgNode()
_CONSTRUCT_LOSS.MARGIN = 1.

LOSS_DEFAULTS = {'ContrastiveLoss': _CONSTRUCT_LOSS}
