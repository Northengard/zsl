import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


def wrap_dataset(dataset, config, is_train, collate_fn=None):
    workers = config.SYSTEM.WORKERS
    if is_train:
        batch_size = config.TRAIN.BATCH_SIZE
    else:
        batch_size = config.TEST.BATCH_SIZE
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            shuffle=is_train,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=collate_fn)
    return dataloader


def support_dataset(data_dir):
    dataset = SupportVector(source_dir=data_dir, transformations=torchvision.transforms.ToTensor())
    return dataset


class SupportVector(Dataset):
    def __init__(self, source_dir, transformations=None):
        self._labels = os.listdir(source_dir)
        self._image_list = [os.path.join(source_dir, img_name) for img_name in self._labels]
        self._labels = [img_name.split('_')[0] for img_name in self._labels]
        self._unique_labels = np.unique(self._labels, return_counts=False)
        self._len = len(self._image_list)
        self._transformations = transformations

    def __len__(self):
        return self._len

    @property
    def transformations(self):
        return self._transformations

    def get_label(self, index):
        return self._labels[index]

    def __getitem__(self, idx):
        img = cv2.imread(self._image_list[idx])
        label = self._labels[idx]
        if self._transformations:
            img = self._transformations(img)
        sample = {'image': img, 'label': label}
        return sample
