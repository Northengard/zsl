import json
from tqdm import tqdm

import cv2
import numpy as np
import torch

from utils import AverageMeter
from utils.visualisation import show_image
from utils.postprocessing import rescale_coords


def get_label_vs_support(sup_matr, nn_output, threshold=None):
    cos_sim = torch.cosine_similarity(sup_matr, nn_output, dim=1)
    pred_label = cos_sim.argmax().item()
    if threshold:
        if cos_sim[pred_label] < threshold:
            return -1, cos_sim[pred_label]
    return pred_label, cos_sim[pred_label]


def evaluation(config, model, dataloader, device):
    model.eval()
    num_iter = len(dataloader)
    num_classes = dataloader.dataset.num_classes
    categ_mapper = dict(zip(list(range(1, num_classes + 1)), dataloader.dataset.categories_ids))
    categories = dataloader.dataset.categories
    conf_matr = torch.zeros(num_classes, num_classes)
    tq = tqdm(total=num_iter)
    tq.set_description(f'Evaluation:')
    loss_handler = AverageMeter()
    boxes_list = list()
    with torch.no_grad():
        for itr, sample in enumerate(dataloader):
            real_image, img_dict, img_id = dataloader.dataset.get_image(itr, get_meta=True)
            images = sample['image']
            images = [image.to(device) for image in images]
            targets = sample['targets']

            # gt = real_image.copy()
            # for obj_id, (box, label_id) in enumerate(zip(targets[0]['boxes'].numpy(),
            #                                              targets[0]['labels'].numpy())):
            #     left, top, right, bottom = box.astype(int)
            #     gt = cv2.rectangle(gt, pt1=(left, top),
            #                        pt2=(right, bottom),
            #                        color=(0, 255, 0),
            #                        thickness=1)
            #
            #     gt = cv2.putText(gt, f'{categories[categ_mapper[label_id]]}',
            #                      org=(left, top - 20),
            #                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                      fontScale=1,
            #                      color=(0, 144, 255),
            #                      thickness=3)
            #
            # show_image(gt, f'gt_img_with_box', wk=0)

            targets = [{cat_id: bboxes.to(device) for cat_id, bboxes in target.items()} for target in targets]

            predicts, _ = model(images, targets)
            predicts = predicts[0]
            for obj_id, (box, label_id, score) in enumerate(zip(predicts['boxes'],
                                                                predicts['labels'],
                                                                predicts['scores'])):
                box = box.cpu().numpy()
                box = rescale_coords(box, images[0].shape[-2:], real_image.shape[:2])
                left, top, right, bottom = box.astype(int)
                box = box.tolist()
                label_id = label_id.cpu().item()
                score = score.cpu().item()
                boxes_list.append({'image_id': img_id, 'bbox': box,
                                   'category_id': categ_mapper[label_id], 'score': score})
                if config.TEST.VISUALIZE:
                    real_image = cv2.rectangle(real_image, pt1=(left, top),
                                               pt2=(right, bottom),
                                               color=(0, 255, 0),
                                               thickness=1)

                    real_image = cv2.putText(real_image, f'{categories[categ_mapper[label_id]]}_{obj_id}',
                                             org=(left, top - 20),
                                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                             fontScale=1,
                                             color=(0, 144, 255),
                                             thickness=3)
            if config.TEST.VISUALIZE:
                show_image(real_image, f'img_with_box', wk=0)
            tq.update(1)
            tq.set_postfix(avg_loss=loss_handler.avg)
    tq.close()

    with open('predictions_topdown.json', 'w') as dumpf:
        json.dump(boxes_list, dumpf)

    return conf_matr.numpy()
