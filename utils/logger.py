import os
import logging
from pathlib import Path
from time import strftime


def get_logger(cfg, cfg_name, phase='train'):
    """
    create logfile and return logger by given log path using logging lib
    :param cfg: experinet config
    :param cfg_name: config file name or path with name
    :param phase: experiment phase name
    :return: logger
    """
    # create logger for prd_ci
    dataset = cfg.DATASET.NAME
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    weight_save_dir = Path(os.path.join(cfg.SYSTEM.SNAPSHOT_DIR, dataset, model, cfg_name))
    weight_save_dir.mkdir(parents=True, exist_ok=True)

    logger_path = os.path.join(cfg.SYSTEM.LOG_DIR, dataset, model, cfg_name)
    timestamp = strftime('%Y-%m-%d-%H-%M')
    logger_name = "_".join([cfg_name, timestamp, phase])+".log"
    logger_path = Path(logger_path)
    logger_path.mkdir(parents=True, exist_ok=True)

    str_prefix = "%(asctime)-15s => %(message)s"
    # create file handler for logger.
    logging.basicConfig(filename=logger_path / logger_name,
                        format=str_prefix)
    log = logging.getLogger()
    log.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    log.addHandler(console)

    log.info(f"snapshots dir: {str(weight_save_dir)}")
    log.info(f"tensorboard log dir: {str(logger_path)}")

    return log, str(weight_save_dir), str(logger_path)
