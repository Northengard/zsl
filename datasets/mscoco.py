import os
import logging
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO

from torch.utils.data import Dataset

from .transformations import Transforms, get_pad, AlbumTransforms
from .common import wrap_dataset


logger = logging.getLogger(__name__)


def collate_fn(batch):
    batch = {key: [sample[key] for sample in batch] for key in batch[0]}
    return batch


def mscoco(config, is_train):
    dataset = MsCocoDataset(config=config, is_train=is_train)
    dataloader = wrap_dataset(dataset=dataset, config=config, is_train=is_train, collate_fn=collate_fn)
    return dataloader


def annot_check(annots):
    return [obj for obj in annots if type(obj['segmentation']) == list]


def check_areas(annots, img_area):
    for obj in annots:
        if round(obj['area']) / img_area > 0.9:
            return False
    return True


class MsCocoDataset(Dataset):
    def __init__(self, config, is_train):
        self._sample_image_hw = config.DATASET.PARAMS.IMAGE_SIZE[::-1]
        self._use_rgb = config.DATASET.PARAMS.USE_RGB

        self._data_dir = config.DATASET.PARAMS.DATA_PATH
        self._data_type = 'train2017' if is_train else 'val2017'
        self._img_path = os.path.join(self._data_dir, self._data_type)
        ann_file = os.path.join(f'{self._data_dir}', 'annotations', f'instances_{self._data_type}.json')
        self._coco_api = COCO(ann_file)

        self._categories = self._coco_api.loadCats(self._coco_api.getCatIds())
        if len(config.DATASET.PARAMS.SUPERCATEG) > 0:
            self._categories = [categ_config for categ_config in self._categories
                                if categ_config['supercategory'] in config.DATASET.PARAMS.SUPERCATEG]
        if len(config.DATASET.PARAMS.CATEG) > 0:
            self._categories = [categ_config for categ_config in self._categories
                                if categ_config['name'] in config.DATASET.PARAMS.CATEG]
        self._categories_repr = {categ['id']: categ['name'] for categ in self._categories}
        self._categories_ids = list(self._categories_repr.keys())
        self._cat_maper = dict(zip(self._categories_ids, list(range(1, len(self._categories_ids) + 1))))

        self._img_id_vs_annot_dict = self._coco_api.imgToAnns

        if not config.TRANSFORMATIONS.USE_ALB:
            self.transforms = Transforms(conf=config, is_train=is_train)
        else:
            self.transforms = AlbumTransforms(conf=config)

        # clear categories ids
        self._img_id_vs_annot_dict = {img_id: self._check_annot_categ(img_annot)
                                      for img_id, img_annot in self._img_id_vs_annot_dict.items()}
        self._img_id_vs_annot_dict = {img_id: img_annot for img_id, img_annot in self._img_id_vs_annot_dict.items()
                                      if len(img_annot) > 0}

        # clear annotations without segmentations
        self._img_id_vs_annot_dict = {img_id: annot_check(img_annot)
                                      for img_id, img_annot in self._img_id_vs_annot_dict.items()}
        self._img_id_vs_annot_dict = {img_id: img_annot for img_id, img_annot in self._img_id_vs_annot_dict.items()
                                      if len(img_annot) > 0}
        self._indexes = list(self._img_id_vs_annot_dict.keys())
        areas = {img_id: self._coco_api.loadImgs(img_id)[0]['height'] * self._coco_api.loadImgs(img_id)[0]['width']
                 for img_id in self._indexes}
        self._img_id_vs_annot_dict = {img_id: img_annot for img_id, img_annot in self._img_id_vs_annot_dict.items()
                                      if check_areas(img_annot, areas[img_id])}
        self._indexes = list(self._img_id_vs_annot_dict.keys())
        self._len = len(self._img_id_vs_annot_dict)

    def __len__(self):
        return self._len

    @property
    def categories_ids(self):
        return self._categories_ids

    @property
    def categories(self):
        return self._categories_repr

    @property
    def num_classes(self):
        return len(self._categories_ids)

    def get_image(self, idx, get_meta=False):
        img_id = self._indexes[idx]
        # self._coco_api.loadImgs returns list of images but only one image is required,
        # so unpack it form list with one element
        # img_dict comp: 'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'
        img_dict = self._coco_api.loadImgs(img_id)[0]

        image = cv2.imread(os.path.join(self._img_path, img_dict['file_name']))
        if self._use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if get_meta:
            return image, img_dict, img_id
        else:
            return image

    def _check_annot_categ(self, img_annot):
        return [obj for obj in img_annot if obj['category_id'] in self._categories_ids]

    def _get_seg_map(self, img_hw, img_annotations):
        semantic_map = np.zeros(self._sample_image_hw)
        _, pad_h, _, pad_w = get_pad(img_h=img_hw[0], img_w=img_hw[1],
                                     nn_h=self._sample_image_hw[0], nn_w=self._sample_image_hw[1])
        scale_coef = self._sample_image_hw[0] / (img_hw[0] + pad_h)
        for obj in img_annotations:
            contours = obj['segmentation']
            for contour in contours:
                mask = np.zeros(self._sample_image_hw)
                contour = np.array(contour).reshape(1, -1, 2)
                contour *= scale_coef
                contour = contour.round().astype(int)
                mask = cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1).astype(bool)
                semantic_map[mask] = self._cat_maper[obj['category_id']]
        logger.debug(f'{np.unique(semantic_map).shape[0] == 1}')
        if np.unique(semantic_map).shape[0] == 1:
            raise ValueError(f'{img_annotations}')
        return semantic_map

    def __getitem__(self, idx):
        image, img_dict, img_id = self.get_image(idx, get_meta=True)

        # self._indexes = list(self._img_id_vs_annot_dict.keys())
        # img = self.get_image(self._indexes.index(363942))
        # logger.debug(f'img.shape: {img.shape}')
        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        # img_annotations: list of dicts
        # with keys: 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
        img_annotations = self._img_id_vs_annot_dict[img_id]
        bboxes = np.array([obj_ann['bbox'] for obj_ann in img_annotations])
        bboxes[:, 2:] += bboxes[:, :2]
        # clear objects with bad boxes
        obj_mask = (bboxes[:, 2:] < bboxes[:, :2]).sum(1).astype(bool)
        bboxes = bboxes[~obj_mask]
        category_ids = np.array([self._cat_maper[obj_ann['category_id']]
                                 for obj_id, obj_ann in enumerate(img_annotations) if not obj_mask[obj_id]])
        seg_masks = self._get_seg_map(image.shape[:-1], img_annotations)
        # 'bbox': bboxes, 'category_id': category_ids,
        sample = {'image': image, 'image_labels': seg_masks, 'bbox': bboxes}
        sample = self.transforms(sample)
        sample['targets'] = {'boxes': sample['bbox'], 'labels': torch.from_numpy(category_ids).to(torch.int64)}
        sample.pop('bbox')
        sample['idx'] = idx
        return sample
