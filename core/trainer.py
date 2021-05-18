import torch
from tqdm import tqdm
from utils import AverageMeter, get_learning_rate
from metrics import get_confusion_matix, prec_rec_acc


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
        # labels = labels.to(device)

        if cfg.TRAIN.IN_MODELL_LOSS:
            targets = batch['targets']
            targets = [{cat_id: bboxes.to(device) for cat_id, bboxes in target.items()} for target in targets]
            output, loss = model(images, targets)
        else:
            image_labels = batch['image_labels'].to(device)
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
        tq.update(cfg.TRAIN.BATCH_SIZE)
    tq.close()


def validation(model, dataloader, loss_fn, device, epoch, cfg):
    model.eval()
    num_iter = len(dataloader)
    tq = tqdm(total=num_iter * cfg.TEST.BATCH_SIZE)
    tq.set_description(f'Validation: Epoch {epoch}')
    loss_handler = AverageMeter()
    conf_matr = torch.zeros(2, 2)
    with torch.no_grad():
        for itr, batch in enumerate(dataloader):
            images = batch['image']
            images = images.to(device)
            image_labels = batch['image_labels'].to(device)

            output = model(images)

            loss = loss_fn(output, image_labels)
            loss_handler.update(loss.item())

            # conf_matr += get_confusion_matix(output, labels)
            tq.update(cfg.TEST.BATCH_SIZE)
            tq.set_postfix(avg_loss=loss_handler.avg)
    tq.close()
    val_results = {cfg.LOSS.NAME: loss_handler.avg}
    val_results.update(prec_rec_acc(conf_matr))
    return val_results
