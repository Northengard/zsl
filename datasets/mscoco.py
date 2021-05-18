import os
import numpy as np
import cv2
from pycocotools.coco import COCO

from torch.utils.data import Dataset

from .transformations import Transforms, get_pad
from .common import wrap_dataset


def mscoco(config, is_train):
    dataset = MsCocoDataset(config=config, is_train=is_train)
    dataloader = wrap_dataset(dataset=dataset, config=config, is_train=is_train)
    return dataloader


def annot_check(annots):
    return [obj for obj in annots if type(obj['segmentation']) == list]


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

        self._img_id_vs_annot_dict = self._coco_api.imgToAnns
        self.transforms = Transforms(conf=config, is_train=is_train)

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

    def get_image(self, idx, return_meta=False):
        img_id = self._indexes[idx]
        # self._coco_api.loadImgs returns list of images but only one image is required,
        # so unpack it form list with one element
        # img_dict comp: 'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'
        img_dict = self._coco_api.loadImgs(img_id)[0]

        image = cv2.imread(os.path.join(self._img_path, img_dict['file_name']))
        if self._use_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if return_meta:
            return image, img_dict, img_id
        else:
            return image

    def _check_annot_categ(self, img_annot):
        return [obj for obj in img_annot if obj['category_id'] in self._categories_ids]

    def _get_seg_map(self, img_hw, img_annotations):
        semantic_map = np.zeros(self._sample_image_hw)
        instance_map = np.zeros(self._sample_image_hw)
        _, pad_h, _, pad_w = get_pad(img_h=img_hw[0], img_w=img_hw[1],
                                     nn_h=self._sample_image_hw[0], nn_w=self._sample_image_hw[1])
        scale_coef = self._sample_image_hw[0] / (img_hw[0] + pad_h)
        instance_id = 1.
        for obj in img_annotations:
            contours = obj['segmentation']
            for contour in contours:
                mask = np.zeros(self._sample_image_hw)
                contour = np.array(contour).reshape(1, -1, 2)
                contour *= scale_coef
                contour = contour.round().astype(int)
                mask = cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1).astype(bool)
                semantic_map[mask] = obj['category_id']
                instance_map[mask] = instance_id
                instance_id += 1
        return np.stack([semantic_map, instance_map], 2)

    def __getitem__(self, idx):
        image, img_dict, img_id = self.get_image(idx, return_meta=True)

        # img_annotations: list of dicts
        # with keys: 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
        img_annotations = self._img_id_vs_annot_dict[img_id]
        bboxes = np.array([obj_ann['bbox'] for obj_ann in img_annotations])
        category_ids = np.array([obj_ann['category_id'] for obj_ann in img_annotations])
        seg_masks = self._get_seg_map(image.shape[:-1], img_annotations)
        # 'bbox': bboxes, 'category_id': category_ids,
        sample = {'image': image, 'image_labels': seg_masks}
        sample = self.transforms(sample)
        sample['idx'] = idx
        return sample
