import time
import torch
from tqdm import tqdm
from utils import AverageMeter, get_learning_rate
from metrics.intersection import get_confusion_matrix, get_iou_metrics
from utils.postprocessing import BottomUpPostprocessing


def train(model, dataloader, loss_fn, optimizer, sheduler, device, logger, board_writer, epoch, cfg):
    logger.info(f'start to train {epoch}')
    loss_handler = AverageMeter()

    num_iter = len(dataloader)
    iters_ahead = epoch * num_iter
    tq = tqdm(total=num_iter * cfg.TRAIN.BATCH_SIZE)
    tq.set_description(f'Train: Epoch {epoch}, lr {get_learning_rate(optimizer):.4e}')

    model.train()
    for itr, batch in enumerate(dataloader):
        images = batch['image']
        images = [image.to(device) for image in images]

        if cfg.TRAIN.IN_MODELL_LOSS:
            targets = batch['targets']
            targets = [{cat_id: bboxes.to(device) for cat_id, bboxes in target.items()} for target in targets]
            output, loss_dict = model(images, targets)
            loss = torch.stack(list(loss_dict.values()))
            loss[0] *= 0.5
            loss = torch.sum(loss)
        else:
            image_labels = torch.stack(batch['image_labels']).to(device)
            images = torch.stack(images)
            output = model(images)
            loss = loss_fn(output, image_labels)
        loss.backward()
        loss_handler.update(loss.item())
        if (itr + 1) % cfg.TRAIN.UPDATE_STEP == 0:
            optimizer.step()
            optimizer.zero_grad()
        if (itr + 1) % cfg.SYSTEM.PRINT_FREQ == 0:
            logger.info(f'iteration {itr + iters_ahead}: avg_loss {loss_handler.avg}')

        board_writer.add_scalar('train/loss', loss.item(), itr + iters_ahead)
        board_writer.add_scalar('train/learning_rate', get_learning_rate(optimizer), itr + iters_ahead)

        if sheduler:
            sheduler.step()
        tq.set_postfix(avg_loss=loss_handler.avg)
        tq.update(cfg.TRAIN.BATCH_SIZE)
    tq.close()
    return loss_handler.avg


def validation(model, dataloader, loss_fn, device, epoch, cfg):
    model.eval()
    num_iter = len(dataloader)
    tq = tqdm(total=num_iter * cfg.TEST.BATCH_SIZE)
    tq.set_description(f'Validation: Epoch {epoch}')
    loss_handler = AverageMeter()
    num_class = dataloader.dataset.num_classes
    with torch.no_grad():
        postproc = BottomUpPostprocessing(classes_ids=dataloader.dataset.categories_ids,
                                          delta=cfg.LOSS.PARAMS.DELTA_V, embedding_size=cfg.MODEL.PARAMS.VECTOR_SIZE,
                                          device=device)
        conf_matr = torch.zeros(num_class + 1, num_class + 1, device=device)
        for itr, batch in enumerate(dataloader):
            images = batch['image']
            images = torch.stack(images).to(device)
            image_labels = torch.stack(batch['image_labels']).to(device)

            output = model(images)

            loss = loss_fn(output, image_labels)
            loss_handler.update(loss.item())

            output = postproc(model_out=output, true_labels=image_labels,
                              ret_cls_pos=False, get_bbox=False, method=0)

            conf_matr += get_confusion_matrix(output, image_labels, num_class)
            tq.update(cfg.TEST.BATCH_SIZE)
            tq.set_postfix(avg_loss=loss_handler.avg)
    tq.close()
    val_results = {cfg.LOSS.NAME: loss_handler.avg}
    val_results.update(get_iou_metrics(conf_matr.cpu().numpy()))
    return val_results
