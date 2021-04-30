
from typing import Tuple, List
import random
import torch
from torch import nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# TODO caching?
from scopeV2.augmentation import Augmentation
from scopeV2.preprocessing import min_max_normalize, resize_volume


class GenevaStrokeOutcomeDataset(Dataset):
    def __init__(self,
                 data_file_path: str,
                 label_file_path: str,
                 outcome: str,
                 channels: List[int] = [0, 1, 2, 3],
                 validation_size: float = 0.3,
                 is_validation: bool = False,
                 neg_to_pos_ratio: int = 0,
                 augmentation_dict={},
                 ):

        # preprocessing parameters
        self.preprocessing_min = 0
        self.preprocessing_max = 400
        self.preprocessing_desired_shape = (46, 46, 46)

        self.data_file_path = data_file_path

        params = np.load(self.data_file_path, allow_pickle=True)['params']
        try:
            print('Using channels:', [params.item()['ct_sequences'][channel] for channel in channels])
        except:
            print('Geneva Stroke Dataset (perfusion CT maps) parameters: ', params)

        all_ids = np.load(self.data_file_path, allow_pickle=True)['ids']

        outcomes_df = pd.read_excel(label_file_path)
        all_labels = np.array([outcomes_df.loc[outcomes_df['anonymised_id'] == subj_id, outcome].iloc[0]
                               for subj_id in all_ids])

        # TODO more efficient loading with np.load(filename, mmap_mode='r') and caching
        raw_images = np.load(self.data_file_path, allow_pickle=True)['ct_inputs'][..., channels]

        # ensure images have a channel dimension
        if raw_images.ndim < 5:
            raw_images = np.expand_dims(raw_images, axis=-1)

        # Apply masks
        raw_masks = np.load(self.data_file_path, allow_pickle=True)['brain_masks']
        raw_masks = np.expand_dims(raw_masks, axis=-1)
        images = raw_images * raw_masks
        all_images = np.array([self.image_preprocessing(image) for image in images])

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
            self.images = all_images[val_indices]
            self.labels = all_labels[val_indices]
            self.ids = all_ids[val_indices]
            self.split_indices = val_indices
        else:
            self.images = all_images[train_indices]
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
        image_t = image_t.permute(3, 0, 1, 2)

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

    def image_preprocessing(self, volume):
        volume = min_max_normalize(volume, self.preprocessing_min, self.preprocessing_max)
        # Resize width, height and depth
        volume = resize_volume(volume, self.preprocessing_desired_shape[0], self.preprocessing_desired_shape[1],
                               self.preprocessing_desired_shape[2])
        return volume
