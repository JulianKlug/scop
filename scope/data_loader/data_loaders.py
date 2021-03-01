from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from base import BaseDataLoader

from data_loader.data_sets.geneva_stroke_outcome_dataset import GenevaStrokeOutcomeDataset

from data_loader.data_sets.geneva_stroke_dataset_pCT import GenevaStrokeDataset_pCT
from data_loader.transformation_sequences import gsd_pCT_train_transform, gsd_pCT_valid_transform


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class GsdOutcomeDataLoader(BaseDataLoader):
    def __init__(self, imaging_dataset_path, outcome_file_path, channels, outcome, preload_data, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = None
        self.dataset = GenevaStrokeOutcomeDataset(imaging_dataset_path, outcome_file_path, channels, outcome, transform=trsfm, preload_data=preload_data)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class GsdOutcomeDataLoader2(DataLoader):
    def __init__(self, imaging_dataset_path, outcome_file_path, channels, outcome, preload_data, batch_size, shuffle=True, validation_split=0.0, num_workers=1):

        train_transform = gsd_pCT_train_transform()
        val_transform = gsd_pCT_valid_transform()

        self.train_dataset = GenevaStrokeDataset_pCT(imaging_dataset_path, outcome_file_path, outcome, split='train',
                                                     transform=train_transform, preload_data=preload_data,
                                                     channels=channels, train_size=0.7, test_size=0.01, valid_size=0.29)

        self.valid_dataset = GenevaStrokeDataset_pCT(imaging_dataset_path, outcome_file_path, outcome, split='validation',
                                                     transform=val_transform, preload_data=preload_data,
                                                     channels=channels, train_size=0.7, test_size=0.01, valid_size=0.29)

        self.valid_loader = DataLoader(dataset=self.valid_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

        super().__init__(dataset=self.train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    def split_validation(self):
        return self.valid_loader
