from typing import Tuple
import random
import torch
from torch import nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


# TODO caching?
from archives.scopeV2.augmentation import Augmentation


class LeftRightDataset(Dataset):
    def __init__(self,
                 data_file_path: str,
                 label_file_path: str,
                 validation_size: float = 0.3,
                 is_validation: bool = False,
                 neg_to_pos_ratio: int = 0,
                 augmentation_dict={},
                 ):

        self.data_file_path = data_file_path
        outcomes_df = pd.read_csv(label_file_path, index_col=0)
        all_labels = np.array(outcomes_df).squeeze()
        all_ids = np.array(range(outcomes_df.shape[0]))

        all_indices = np.array(range(len(all_ids)))

        if validation_size == 0:
            train_indices = all_indices
            random.Random(42).shuffle(train_indices)
        else:
            train_indices, val_indices = train_test_split(all_indices, test_size=validation_size, random_state=42,
                                                          shuffle=True, stratify=all_labels)

        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.augmentation_dict = augmentation_dict

        self.augmentation = Augmentation(**self.augmentation_dict)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.augmentation = nn.DataParallel(self.augmentation)
            self.augmentation = self.augmentation.to(torch.device("cuda"))

        if is_validation:
            self.images = np.load(data_file_path)[val_indices]
            self.labels = all_labels[val_indices]
            self.ids = all_ids[val_indices]
            self.split_indices = val_indices
        else:
            self.images = np.load(data_file_path)[train_indices]
            self.labels = all_labels[train_indices]
            self.ids = all_ids[train_indices]
            self.split_indices = train_indices

        # indices here are referring to the split space
        self.neg_indices = np.where(self.labels == 0)[0]
        self.pos_indices = np.where(self.labels == 1)[0]

    def __len__(self):
        if self.neg_to_pos_ratio:
            return int(len(self.neg_indices) + len(self.neg_indices) / self.neg_to_pos_ratio)
        else:
            return len(self.ids)

    def shuffleSamples(self):
        if self.neg_to_pos_ratio:
            random.Random(42).shuffle(self.neg_indices)
            random.Random(42).shuffle(self.pos_indices)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, int]:
        """Get tuple (image tensor, label tensor, id)"""

        # Perform balancing by selecting a positive sample vs negative sample according to neg_to_pos_ratio
        if self.neg_to_pos_ratio:
            pos_index = index // (self.neg_to_pos_ratio + 1)
            if index % (self.neg_to_pos_ratio + 1):
                neg_index = index - 1 - pos_index
                neg_index %= len(self.neg_indices)
                sample_index = self.neg_indices[neg_index]
            else:
                pos_index %= len(self.pos_indices)
                sample_index = self.pos_indices[pos_index]
        else:
            sample_index = index

        image_a = self.images[sample_index]
        image_t = torch.from_numpy(image_a)
        image_t = image_t.to(torch.float32)
        image_t = image_t.unsqueeze(0)

        if self.augmentation_dict:
            image_t = self.augmentation(image_t)

        # right labels
        label_t = torch.tensor([
            not self.labels[sample_index],
            self.labels[sample_index]
        ], dtype=torch.long)

        return (
            image_t,
            label_t,
            self.ids[sample_index]
        )
