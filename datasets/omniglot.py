import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np


def omniglot(config, is_train):
    dataset = ZSLOmniglot(is_train=is_train,
                          pos_neg=config.DATASET.PARAMS.POS_NEG_RATIO,
                          test_langs=config.DATASET.PARAMS.TEST_LANGS_LIST,
                          download=config.DATASET.PARAMS.DOWNLOAD)
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
                            drop_last=True)
    return dataloader


class ZSLOmniglot(Dataset):
    def __init__(self, is_train, pos_neg, test_langs, download):
        self._alphabet_vs_letter = {'Alphabet_of_the_Magi': 20,
                                    'Anglo-Saxon_Futhorc': 29,
                                    'Arcadian': 26,
                                    'Armenian': 41,
                                    'Asomtavruli_(Georgian)': 40,
                                    'Balinese': 24,
                                    'Bengali': 46,
                                    'Blackfoot_(Canadian_Aboriginal_Syllabics)': 14,
                                    'Braille': 26,
                                    'Burmese_(Myanmar)': 34,
                                    'Cyrillic': 33,
                                    'Early_Aramaic': 22,
                                    'Futurama': 26,
                                    'Grantha': 43,
                                    'Greek': 24,
                                    'Gujarati': 48,
                                    'Hebrew': 22,
                                    'Inuktitut_(Canadian_Aboriginal_Syllabics)': 16,
                                    'Japanese_(hiragana)': 52,
                                    'Japanese_(katakana)': 47,
                                    'Korean': 40,
                                    'Latin': 26,
                                    'Malay_(Jawi_-_Arabic)': 40,
                                    'Mkhedruli_(Georgian)': 41,
                                    'N_Ko': 33,
                                    'Ojibwe_(Canadian_Aboriginal_Syllabics)': 14,
                                    'Sanskrit': 42,
                                    'Syriac_(Estrangelo)': 23,
                                    'Tagalog': 17,
                                    'Tifinagh': 55}
        self.letter_amount = 20
        self._alphabets_samples_range = {key: val * self.letter_amount for key, val in self._alphabet_vs_letter.items()}
        self._alphabets_samples_range = torch.tensor(list(self._alphabets_samples_range.values()))
        self._alphabets_samples_range = (torch.cumsum(self._alphabets_samples_range, 0) - 1).tolist()
        self._alphabets_samples_range.insert(0, -1)
        self._alphabets_samples_range = [[self._alphabets_samples_range[idx - 1] + 1,
                                          self._alphabets_samples_range[idx]]
                                         for idx in range(1, len(self._alphabets_samples_range))]
        self._alphabets_samples_range = dict(zip(self._alphabet_vs_letter.keys(), self._alphabets_samples_range))

        self._alphabet_vs_letter = {key: val for key, val in self._alphabet_vs_letter.items()
                                    if is_train != (key in test_langs)}
        self._alphabets_samples_range = {key: val for key, val in self._alphabets_samples_range.items()
                                         if is_train != (key in test_langs)}

        self._data_source = torchvision.datasets.Omniglot(root="./data",
                                                          download=download,
                                                          transform=torchvision.transforms.ToTensor())

        self.data_amount = sum(self._alphabet_vs_letter.values()) * self.letter_amount
        self.pos_neg = pos_neg
        self._keeped_indexes = [list(range(min_idx, max_idx + 1))
                                for min_idx, max_idx in self._alphabets_samples_range.values()]
        self._keeped_indexes = [idx for idx_set in self._keeped_indexes for idx in idx_set]

    def __len__(self):
        return self.data_amount

    def __getitem__(self, sub_idx):
        real_idx = self._keeped_indexes[sub_idx]
        # position inside 20 same letters
        # its relative index, same for outer idx
        current_sample_id = (real_idx % self.letter_amount)
        low = sub_idx - current_sample_id
        up = low + self.letter_amount
        if torch.rand(1) < self.pos_neg:
            choice_list = self._keeped_indexes[low:up]
            choice_list.pop(current_sample_id)
        else:
            choice_list = self._keeped_indexes[:low] + self._keeped_indexes[up:]
        another_sample_idx = np.random.choice(choice_list)
        images = list()
        labels = list()
        for sub_idx in [real_idx, another_sample_idx]:
            image, label = self._data_source[sub_idx]
            images.append(image)
            labels.append(label)

        input_tensor = torch.stack(images, 0)
        input_label = torch.tensor(float(labels[0] == labels[1]), dtype=torch.float32)

        return {'image': input_tensor, 'label': input_label, 'image_labels': labels}
