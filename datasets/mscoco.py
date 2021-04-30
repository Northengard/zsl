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

        self._data_dir = os.path.sep.join(['/storage', 'Datasets', 'mscoco'])
        self._data_type = 'train2017' if is_train else 'val2017'
        self._img_path = os.path.join(self._data_dir, self._data_type)
        ann_file = os.path.join(f'{self._data_dir}', 'annotations', f'instances_{self._data_type}.json')
        self._coco_api = COCO(ann_file)

        self._categories = self._coco_api.loadCats(self._coco_api.getCatIds())

        self._img_id_vs_annot_dict = self._coco_api.imgToAnns
        self._len = len(self._img_id_vs_annot_dict)
        self.transforms = Transforms(conf=config, is_train=is_train)
        self._img_id_vs_annot_dict = {img_id: annot_check(img_annot)
                                      for img_id, img_annot in self._img_id_vs_annot_dict.items()}
        self._img_id_vs_annot_dict = {img_id: img_annot for img_id, img_annot in self._img_id_vs_annot_dict.items()
                                      if len(img_annot) > 0}
        self._indexes = list(self._img_id_vs_annot_dict.keys())

    def __len__(self):
        return self._len

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
        img_id = self._indexes[idx]
        # self._coco_api.loadImgs returns list of images but only one image is required,
        # so unpack it form list with one element
        # img_dict comp: 'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'
        img_dict = self._coco_api.loadImgs(img_id)[0]

        image = cv2.imread(os.path.join(self._img_path, img_dict['file_name']))
        # img_annotations: list of dicts
        # with keys: 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
        img_annotations = self._img_id_vs_annot_dict[img_id]
        bboxes = np.array([obj_ann['bbox'] for obj_ann in img_annotations])
        category_ids = np.array([obj_ann['category_id'] for obj_ann in img_annotations])
        seg_masks = self._get_seg_map(image.shape[:-1], img_annotations)
        # 'bbox': bboxes, 'category_id': category_ids,
        sample = {'image': image, 'image_labels': seg_masks}
        sample = self.transforms(sample)
        return sample
