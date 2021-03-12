from yacs.config import CfgNode

_BASE_ZSL = CfgNode()
_BASE_ZSL.N_BLOCKS = 4
_BASE_ZSL.START_OUT_CHN = 8
_BASE_ZSL.CHN_SCALE_COEF = 2

MODEL_DEFAULTS = {'base_model': _BASE_ZSL}
