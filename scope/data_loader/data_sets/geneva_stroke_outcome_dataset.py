import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class GenevaStrokeOutcomeDataset(Dataset):
    """Geneva Clinical Stroke Outcome Dataset."""

    def __init__(self, imaging_dataset_path, outcome_file_path, channels, outcome, transform=None, preload_data=True):
        """

        """

        self.imaging_dataset_path = imaging_dataset_path
        self.params = np.load(imaging_dataset_path, allow_pickle=True)['params']
        self.channels = channels
        self.ids = np.load(imaging_dataset_path, allow_pickle=True)['ids']

        outcomes_df = pd.read_excel(outcome_file_path)
        raw_labels = [outcomes_df.loc[outcomes_df['anonymised_id'] == subj_id, outcome].iloc[0] for
                      subj_id in self.ids]
        self.raw_labels = torch.tensor(raw_labels).long()

        try:
            print('Using channels:', [self.params.item()['ct_sequences'][channel] for channel in channels])
        except:
            print('Geneva Stroke Dataset (perfusion CT maps) parameters: ', self.params)


        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data

        if self.preload_data:
            print('Preloading the dataset ...')
            # select only from data available for this split
            self.raw_images = np.load(imaging_dataset_path, allow_pickle=True)['ct_inputs'][..., self.channels]
            self.raw_masks = np.load(imaging_dataset_path, allow_pickle=True)['brain_masks']

            self.raw_masks = np.expand_dims(self.raw_masks, axis=-1)
            if self.raw_images.ndim < 5:
                self.raw_images = np.expand_dims(self.raw_images, axis=-1)

            # Apply masks
            self.raw_images = self.raw_images * self.raw_masks

            assert len(self.raw_images) == len(self.raw_labels)
            print('Loading is done\n')

    def get_ids(self, indices):
        return [self.ids[index] for index in indices]

    def __getitem__(self, index):
        '''
        Return sample at index
        :param index: int
        :return: sample (x, y, z, c)
        '''

        # load the images
        if not self.preload_data:
            input = np.load(self.imaging_dataset_path, allow_pickle=True)['ct_inputs'][index, ..., self.channels]

            mask = np.load(self.imaging_dataset_path, allow_pickle=True)['brain_masks'][index]

            # Make sure there is a channel dimension
            mask = np.expand_dims(mask, axis=-1)
            if input.ndim < 5:
                input = np.expand_dims(input, axis=-1)

            # Apply masks
            input = input * mask
            # Remove first dimension
            input = np.squeeze(input, axis=0)


        else:
            # With preload, it is already only the images from a certain split that are loaded
            input = self.raw_images[index]

        target = self.raw_labels[index]
        id = self.ids[index]

        input = torch.tensor(input).permute(3, 0, 1, 2).to(torch.float32)

        # apply transformations
        if self.transform:
            input = self.transform(input)

        return input, target, id

    def __len__(self):
        return len(self.ids)