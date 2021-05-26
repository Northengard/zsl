import os
import argparse
import pprint
import setuptools

from numpy import inf

import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from config import config, update_config
from utils import get_logger
import datasets
import models
import losses
from core import train, validation, evaluation
from utils.storage import save_weights, load_weights
from utils.visualisation import draw_confusion_matrix


def parse_args(arg_list):
    parser = argparse.ArgumentParser('ZSL main')
    parser.add_argument('--cfg', required=True, help='path to config file')
    parser.add_argument('-p', '--phase', type=str, default='train', choices=['train', 'eval'],
                        help='Phase of experiment; set `train` for training end `eval` for evaluation')
    parser.add_argument('--opts', nargs=argparse.REMAINDER,
                        help='Modify config via command-line. Use <attrib_name> <new_val> pairs with whitespace sep')
    args = parser.parse_args(args=arg_list)
    update_config(config, args=args)
    return args


def main_eval(config, model, proc_device, logger, writer, cfg):
    model = model.eval()

    eval_loader = getattr(datasets, cfg.DATASET.NAME)(cfg, is_train=False)
    # support_dataset = getattr(datasets, 'support_dataset')(cfg.TEST.DATA_SOURCE)
    # classes = eval_loader.dataset.categories
    # support_matrix = list()
    # sup_labels = ['None']
    # with torch.no_grad():
    #     for item in support_dataset:
    #         sup_image = item['image'].to(proc_device)
    #         sup_labels.append(item['label'])
    #         support_matrix.append(model(sup_image.unsqueeze(0)))
    # support_matrix = torch.stack(support_matrix, 0)
    conf_matr = evaluation(config=config, model=model, dataloader=eval_loader, device=proc_device)
    logger.info('confusion_matrix')
    logger.info(conf_matr)
    # writer.add_figure('eval conf_matrix', draw_confusion_matrix(conf_matr, sup_labels))


def main_train(model, proc_device, loss, optimizer, logger, writer, snapshot_dir, cfg, start_epoch):
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
    for epoch in range(start_epoch + 1, cfg.TRAIN.N_EPOCHS):
        avg_loss = train(model=model, dataloader=train_loader,
                         loss_fn=loss, optimizer=optimizer,
                         sheduler=None if cfg.TRAIN.SCHEDULER == 'plateau' else scheduler,
                         device=device, logger=logger, board_writer=writer, epoch=epoch, cfg=cfg)

        is_val_required = cfg.TRAIN.VAL_REQUIRED and (epoch % cfg.TRAIN.VAL_FREQ == 0)
        if is_val_required:
            logger.info(f'start to validate {epoch}')
            val_output = validation(model=model, dataloader=val_loader, device=device,
                                    loss_fn=loss, epoch=epoch, cfg=cfg)
            logger.info(f'validation results:\n{val_output}')
        else:

            logger.info(f'validation skipped, is_val_required = {is_val_required}')
            val_output = {cfg.LOSS.NAME: avg_loss}
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


def main(proc_device, args, cfg):
    is_train = args.phase == 'train'

    logger, snapshot_dir, log_dir = get_logger(cfg=cfg, cfg_name=os.path.basename(args.cfg), phase=args.phase)
    if proc_device == 0:
        writer = SummaryWriter(log_dir=log_dir, max_queue=100, flush_secs=5)
    else:
        writer = None

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(cfg))

    logger.info(f"Using {proc_device} device")

    if cfg.TRAIN.IN_MODELL_LOSS:
        loss = None
    else:
        loss = getattr(losses, cfg.LOSS.NAME)(cfg)

    start_epoch = 0
    model = getattr(models, cfg.MODEL.NAME)(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR) if is_train else None
    if cfg.MODEL.PRETRAINED:
        logger.info('loading: pretrained weights')
        load_weights(model=model,
                     optimizer=optimizer,
                     checkpoint_file=cfg.MODEL.PRETRAINED)
        start_epoch = int(os.path.basename(cfg.MODEL.PRETRAINED).split('.')[0][-1])
    if cfg.SYSTEM.PARALLEL:
        model = DataParallel(model)
    model = model.to(proc_device)

    if is_train:
        main_train(model=model,
                   proc_device=proc_device,
                   loss=loss,
                   optimizer=optimizer,
                   logger=logger,
                   writer=writer,
                   snapshot_dir=snapshot_dir,
                   cfg=cfg,
                   start_epoch=start_epoch)
    else:
        main_eval(config=config,
                  model=model,
                  proc_device=proc_device,
                  logger=logger,
                  writer=writer,
                  cfg=cfg)


if __name__ == '__main__':
    arguments = parse_args(os.sys.argv[1:])
    config.defrost()
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 0:
        config.SYSTEM.NGPUS = torch.cuda.device_count()
        config.SYSTEM.PARALLEL = config.SYSTEM.NGPUS > 1
    if arguments.phase == 'eval':
        config.TEST.BATCH_SIZE = 1
    config.freeze()
    main(proc_device=device, args=arguments, cfg=config)
