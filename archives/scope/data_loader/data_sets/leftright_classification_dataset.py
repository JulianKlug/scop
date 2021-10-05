import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class LeftRightClassificationDataset(Dataset):
    """
    This dataset generates shapes (only cubes for now) with centers distributed along 3D spatial axes. The label
    corresponds their position on the x-axis with:
        label 0: left (xi < half width of x axis)
        label 1: right (xi > half width of x axis)
    """

    def __init__(self, imaging_dataset_path, label_file_path, transform=None, preload_data=True):
        """

        """

        self.imaging_dataset_path = imaging_dataset_path

        outcomes_df = pd.read_csv(label_file_path, index_col=0)
        self.ids = np.array(range(outcomes_df.shape[0]))

        raw_labels = np.array(outcomes_df).squeeze()
        self.raw_labels = torch.tensor(raw_labels).long()

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data

        if self.preload_data:
            print('Preloading the dataset ...')
            self.raw_images = np.load(imaging_dataset_path)

            if self.raw_images.ndim < 5:
                self.raw_images = np.expand_dims(self.raw_images, axis=-1)

            assert len(self.raw_images) == len(self.raw_labels)
            print('Loading is done\n')

    def get_ids(self, indices):
        return [self.ids[index] for index in indices]

    def __getitem__(self, index):
        '''
        Return sample at index
        :param index: int
        :return: sample (c, x, y, z)
        '''

        # load the images
        if not self.preload_data:
            input = np.load(self.imaging_dataset_path)[index]

            if input.ndim < 5:
                input = np.expand_dims(input, axis=-1)

            input = np.squeeze(input, axis=0)

        else:
            # With preload, it is already only the images from a certain split that are loaded
            input = self.raw_images[index]

        target = self.raw_labels[index]
        id = self.ids[index]

        input = np.transpose(input, (3, 0, 1, 2))

        # apply transformations
        if self.transform:
            # transiently transform into dictionary to use DKFZ augmentation
            data_dict = {'data': input}
            input = self.transform(**data_dict)['data']
            input = torch.from_numpy(input).to(torch.float32)

        return input, target, id

    def __len__(self):
        return len(self.ids)
