import os
import argparse
import pprint

from numpy import inf

import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from config import config, update_config
from utils import get_logger
import datasets
import models
import losses
from core import train, validation
from utils.storage import save_weights, load_weights


def parse_args(arg_list):
    parser = argparse.ArgumentParser('ZSL main')
    parser.add_argument('--cfg', required=True, help='path to config file')
    parser.add_argument('--opts', nargs=argparse.REMAINDER,
                        help='Modify config via command-line. Use <attrib_name> <new_val> pairs with whitespace sep')
    args = parser.parse_args(args=arg_list)
    update_config(config, args=args)
    return args


def main(proc_device, args, cfg):
    logger, snapshot_dir, log_dir = get_logger(cfg=cfg, cfg_name=os.path.basename(args.cfg))
    if proc_device == 0:
        writer = SummaryWriter(log_dir=log_dir, max_queue=100, flush_secs=5)
    else:
        writer = None

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(cfg))

    logger.info(f"Using {proc_device} device")

    loss = getattr(losses, cfg.LOSS.NAME)(cfg)

    model = getattr(models, cfg.MODEL.NAME)(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR)
    if cfg.MODEL.PRETRAINED:
        load_weights(model=model,
                     optimizer=optimizer,
                     checkpoint_file=cfg.MODEL.PRETRAINED)
    if cfg.SYSTEM.PARALLEL:
        model = DataParallel(model)
    model = model.to(device)

    if cfg.TRAIN.SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEPS,
                                                         gamma=cfg.TRAIN.LR_FACTOR)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=cfg.TRAIN.LR_FACTOR,
                                                               patience=cfg.TRAIN.LR_STEPS[0],
                                                               min_lr=cfg.TRAIN.LR / len(cfg.TRAIN.LR_STEPS))

    logger.info(f'model {cfg.MODEL.NAME} initialized')

    train_loader = getattr(datasets, cfg.DATASET.NAME)(cfg, is_train=True)
    val_loader = getattr(datasets, cfg.DATASET.NAME)(cfg, is_train=False)
    logger.info(f'dataloaders {cfg.DATASET.NAME} up')

    best_loss = inf
    for epoch in range(cfg.TRAIN.N_EPOCHS):
        train(model=model, dataloader=train_loader,
              loss_fn=loss, optimizer=optimizer,
              sheduler=None if cfg.TRAIN.SCHEDULER == 'plateau' else scheduler,
              device=device, logger=logger, board_writer=writer, epoch=epoch, cfg=cfg)

        logger.info(f'start to validate {epoch}')
        val_output = validation(model=model, dataloader=val_loader, device=device,
                                loss_fn=loss, epoch=epoch, cfg=cfg)
        if cfg.TRAIN.SCHEDULER != 'multistep':
            scheduler.step(val_output[cfg.LOSS.NAME])
        if proc_device == 0:
            for metric_name, metric_val in val_output.items():
                writer.add_scalar(f'validation/{metric_name}', metric_val, epoch * (len(train_loader) + 1))

        if val_output[cfg.LOSS.NAME] < best_loss:
            best_loss = val_output[cfg.LOSS.NAME]
            save_weights(model=model,
                         optimizer=optimizer,
                         prefix='best',
                         model_name=cfg.MODEL.NAME,
                         epoch=epoch,
                         save_dir=snapshot_dir,
                         parallel=cfg.SYSTEM.PARALLEL)
        elif (epoch + 1) % cfg.SYSTEM.SAVE_FREQ == 0:
            save_weights(model=model,
                         optimizer=optimizer,
                         prefix='checkpoint',
                         model_name=cfg.MODEL.NAME,
                         epoch=epoch,
                         save_dir=snapshot_dir,
                         parallel=cfg.SYSTEM.PARALLEL)
    logger.info('Done')


if __name__ == '__main__':
    arguments = parse_args(os.sys.argv[1:])
    config.defrost()
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device:
        config.SYSTEM.NGPUS = torch.cuda.device_count()
        config.SYSTEM.PARALLEL = config.SYSTEM.NGPUS > 1
    config.freeze()
    main(proc_device=device, args=arguments, cfg=config)
