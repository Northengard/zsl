import os
import argparse
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

import datasets
import models
from config import config, update_config
from utils.storage import load_weights
from utils.postprocessing import BottomUpPostprocessing
from utils.visualisation import alpha_blend, show_image
from datasets.transformations import Rescale
from metrics.intersection import get_iou_metrics, get_confusion_matrix


def parse_args(arg_list):
    parser = argparse.ArgumentParser('ZSL main')
    parser.add_argument('--cfg', default='experiments/bottom_up/mscoco_semantic_mobnetV3_upd_discr_bs12_ep20_lr3e-2_plateau.yaml',
                        help='path to config file')
    parser.add_argument('-p', '--phase', type=str, default='train', choices=['train', 'eval'],
                        help='Phase of experiment; set `train` for training end `eval` for evaluation')
    parser.add_argument('--opts', nargs=argparse.REMAINDER,
                        help='Modify config via command-line. Use <attrib_name> <new_val> pairs with whitespace sep')
    args = parser.parse_args(args=arg_list)
    update_config(config, args=args)
    return args


if __name__ == '__main__':
    parse_args(os.sys.argv[1:])
    model = getattr(models, config.MODEL.NAME)(config)
    optimizer = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.MODEL.PRETRAINED:
        load_weights(model=model,
                     optimizer=optimizer,
                     checkpoint_file=config.MODEL.PRETRAINED)

    model = model.to(device)
    model.eval()
    val_loader = getattr(datasets, config.DATASET.NAME)(config, is_train=False)
    postproc = BottomUpPostprocessing(val_loader.dataset.categories_ids, delta=0.75,
                                      embedding_size=config.MODEL.PARAMS.VECTOR_SIZE, device=device)
    rescale = Rescale(config.DATASET.PARAMS.IMAGE_SIZE)
    num_classes = val_loader.dataset.num_classes
    confision_matrix = torch.zeros(num_classes + 1, num_classes + 1, device=device)
    categories = val_loader.dataset.categories
    tq = tqdm(total=len(val_loader))
    # list of dicts with "area","image_id","bbox", "category_id", "id": 36443
    # self._coco_api.loadRes()

    boxes_list = list()
    visualize = False
    with_bbox = True
    with torch.no_grad():
        for itr, batch in enumerate(val_loader):
            real_image, img_dict, img_id = val_loader.dataset.get_image(itr, get_meta=True)
            real_image = rescale({'image': real_image})['image']
            images = torch.stack(batch['image']).to(device)
            labels = torch.stack(batch['image_labels']).to(device)
            semantic = model(images)
            processed_semantic_map, cls_data, scores = postproc(semantic, true_labels=labels, method=0,
                                                                ret_cls_pos=True, get_bbox=with_bbox)
            for class_id, cls_pos in cls_data.items():
                real_cls_id = postproc.get_mapped_category_id(class_id)
                for obj_id, obj_pos in enumerate(cls_pos):
                    if with_bbox:
                        boxes_list.append({'image_id': img_id, 'bbox': obj_pos.tolist(),
                                           'category_id': real_cls_id, 'score': scores[class_id][obj_id]})
                    else:
                        boxes_list.append({'image_id': img_id, 'segmentation': obj_pos.tolist(),
                                           'category_id': real_cls_id, 'score': scores[class_id][obj_id]})
            batch_confision_matrix = get_confusion_matrix(labels,
                                                          processed_semantic_map, num_classes, ignore=-1)
            confision_matrix += batch_confision_matrix
            sample_pix_metrics = get_iou_metrics(batch_confision_matrix.cpu().numpy())
            # visualisation
            if visualize:
                gt_to_vis = labels.cpu().squeeze().numpy().astype('uint8')
                for proc_map in processed_semantic_map:
                    color_map = proc_map.cpu().squeeze().numpy().astype('uint8')
                    values = np.unique(color_map)
                    for val in values:
                        c = ((np.random.random(1, ) * 0.6 + 0.4) * 255).astype('uint8')
                        color_map[color_map == val] = c
                    color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)
                    gt_to_vis = cv2.applyColorMap(gt_to_vis, cv2.COLORMAP_JET)
                    img_to_vis = alpha_blend(real_image, color_map, (color_map > 0) * 128)
                    for class_id, cls_pos in cls_data.items():
                        real_cls_id = postproc.get_mapped_category_id(class_id)
                        for obj_id, obj_pos in enumerate(cls_pos):
                            left, top, right, bottom = obj_pos
                            real_image = cv2.rectangle(img=real_image,
                                                       pt1=(left, top), pt2=(right, bottom),
                                                       color=(0, 255, 0),
                                                       thickness=1)
                            # print(categories[real_cls_id], real_cls_id)
                            real_image = cv2.putText(real_image, f'{categories[real_cls_id]}_{obj_id}',
                                                     org=(left, top - 20),
                                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                     fontScale=1,
                                                     color=(0, 144, 255),
                                                     thickness=3)
                    show_image(img_to_vis, f'blended', wk=1)
                    show_image(color_map, f'colormap', wk=1)
                    show_image(gt_to_vis, f'gt_to_vis', wk=1)
                    show_image(real_image, f'gt_with_box', wk=1)
            tq.set_postfix(sample_pix_metrics)
            tq.update()
            # print(itr)
        tq.close()
        iou_metrics = get_iou_metrics(confision_matrix.cpu().numpy())
        with open('predictions_seg.json', 'w') as dumpf:
            json.dump({'predicts': boxes_list, 'pixel_metrics': iou_metrics}, dumpf)
        print(iou_metrics)
        cv2.destroyAllWindows()
