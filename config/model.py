from yacs.config import CfgNode

_BASE_ZSL = CfgNode()
_BASE_ZSL.N_BLOCKS = 4
_BASE_ZSL.START_OUT_CHN = 8
_BASE_ZSL.CHN_SCALE_COEF = 2
_BASE_ZSL.VECTOR_SIZE = 200

_SEG_RCNN = CfgNode()
_SEG_RCNN.ENCODER = 'mobilenet'
_SEG_RCNN.DECODER_CHANNELS = [512, 320, 128, 64, 32]
_SEG_RCNN.VECTOR_SIZE = _SEG_RCNN.DECODER_CHANNELS[-1]

_TOPDOWN_RCNN = CfgNode()
_TOPDOWN_RCNN.VECTOR_SIZE = 256

MODEL_DEFAULTS = {'base_model': _BASE_ZSL, 'seg_rcnn': _SEG_RCNN, 'topdown_rcnn': _TOPDOWN_RCNN}
