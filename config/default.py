from yacs.config import CfgNode
from .dataset import DATASET_DEFAULTS
from .model import MODEL_DEFAULTS
from .loss import LOSS_DEFAULTS


_C = CfgNode()

_C.SYSTEM = CfgNode()
_C.SYSTEM.LOG_DIR = "logs"
_C.SYSTEM.SNAPSHOT_DIR = 'snapshots'
_C.SYSTEM.WORKERS = 4
_C.SYSTEM.NGPUS = 1
_C.SYSTEM.SAVE_FREQ = 10  # in epoch. In general may be better to operate with iterations.
_C.SYSTEM.PRINT_FREQ = 100
_C.SYSTEM.PARALLEL = False

_C.DATASET = CfgNode()
_C.DATASET.NAME = "omniglot"
_C.DATASET.PARAMS = CfgNode(new_allowed=True)

_C.MODEL = CfgNode()
_C.MODEL.NAME = "base_model"
_C.MODEL.PARAMS = CfgNode(new_allowed=True)
_C.MODEL.PRETRAINED = ''

_C.TRAIN = CfgNode()
_C.TRAIN.N_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 10
_C.TRAIN.SCHEDULER = 'multistep'
_C.TRAIN.LR = 1e-2
_C.TRAIN.LR_FACTOR = 0.2
_C.TRAIN.LR_STEPS = [10000, 20000, 30000]
_C.TRAIN.UPDATE_STEP = 1

_C.TEST = CfgNode()
_C.TEST.BATCH_SIZE = _C.TRAIN.BATCH_SIZE

_C.LOSS = CfgNode()
_C.LOSS.NAME = 'ContrastiveLoss'
_C.LOSS.PARAMS = CfgNode(new_allowed=True)


def update_config(cfg, args):
    cfg.defrost()
    cfg.DATASET.PARAMS = DATASET_DEFAULTS[cfg.DATASET.NAME]
    cfg.MODEL.PARAMS = MODEL_DEFAULTS[cfg.MODEL.NAME]
    cfg.LOSS.PARAMS = LOSS_DEFAULTS[cfg.LOSS.NAME]
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
