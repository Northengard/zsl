import os
import numpy as np
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from transformations import Transforms
from common import wrap_dataset


def mscoco(config, is_train):
    dataset = MsCocoDataset(config=config, is_train=is_train)
    dataloader = wrap_dataset(dataset=dataset, config=config, is_train=is_train)
    return dataloader


class MsCocoDataset(Dataset):
    def __init__(self, config, is_train):
        self._data_dir = os.path.sep.join(['.', 'data', 'ms_coco'])
        self._data_type = 'train2017' if is_train else 'val2017'
        self._img_path = os.path.join(self._data_dir, self._data_type)
        ann_file = os.path.join(f'{self._data_dir}', 'annotations', f'instances_{self._data_type}.json')
        self._coco_api = COCO(ann_file)

        self._categories = self._coco_api.loadCats(self._coco_api.getCatIds())
        # self._categories_names = [cat['name'] for cat in self._categories]
        # print('COCO categories: \n{}\n'.format(' '.join(self._categories_names)))

        self._img_id_vs_annot_dict = self._coco_api.imgToAnns
        self._len = len(self._img_id_vs_annot_dict)
        self._indexes = list(self._img_id_vs_annot_dict.keys())
        self.transforms = Transforms(conf=config, is_train=is_train)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img_id = self._indexes[idx]
        # self._coco_api.loadImgs returns list of images but only one image is required,
        # so unpack it form list with one element
        # img_dict comp: 'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'
        img_dict = self._coco_api.loadImgs(img_id)[0]

        image = os.path.join(self._img_path, img_dict['file_name'])
        # img_annotations: list of dicts
        # with keys: 'segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'
        img_annotations = self._img_id_vs_annot_dict[img_id]
        bboxes = np.array([obj_ann['bbox'] for obj_ann in img_annotations])
        category_ids = np.array([obj_ann['category_id'] for obj_ann in img_annotations])
        sample = {'image': image, 'bbox': bboxes, 'category_id': category_ids}
        sample = self.transforms(sample)
        return sample
