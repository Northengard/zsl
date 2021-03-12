import torch
from tqdm import tqdm
from utils import AverageMeter, get_learning_rate


def train(model, dataloader, loss_fn, optimizer, sheduler, device, logger, board_writer, epoch, cfg):
    logger.info(f'start to train {epoch}')
    loss_handler = AverageMeter()

    num_iter = len(dataloader)
    iters_ahead = epoch * num_iter
    tq = tqdm(total=num_iter * cfg.TRAIN.BATCH_SIZE)
    tq.set_description(f'Train: Epoch {epoch}, lr {get_learning_rate(optimizer):.4e}')

    model.train()
    for itr, sample in enumerate(dataloader):
        images, labels = sample['image'], sample['label']
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = loss_fn(output, labels)
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
        tq.update()
    tq.close()


def validation(model, dataloader, loss_fn, device, epoch, cfg):
    model.eval()
    num_iter = len(dataloader)
    tq = tqdm(total=num_iter * cfg.TRAIN.BATCH_SIZE)
    tq.set_description(f'Validation: Epoch {epoch}')
    loss_handler = AverageMeter()
    for itr, sample in enumerate(dataloader):
        images, labels = sample['image'], sample['label']
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = loss_fn(output, labels)
        loss_handler.update(loss.item())

        tq.update()
        tq.set_postfix(avg_loss=loss_handler.avg)
    tq.close()
    return {'loss': loss_handler.avg}
