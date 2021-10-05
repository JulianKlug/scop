import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 stratify=True, val_dataset=None):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, dataset.raw_labels,
                                                               stratify=stratify)

        if val_dataset is None:
            # this way val dataset can have different transform applied
            val_dataset = dataset

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        self.val_init_kwargs = {
            'dataset': val_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, stratification_labels, stratify=True):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        if stratify:
            train_idx, valid_idx = train_test_split(idx_full, test_size=split, random_state=42,
                                                    shuffle=True, stratify=stratification_labels)
        else:
            train_idx, valid_idx = train_test_split(idx_full, test_size=split, random_state=42,
                                                    shuffle=True)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.val_init_kwargs)
