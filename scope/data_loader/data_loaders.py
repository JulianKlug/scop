from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from base import BaseDataLoader

from data_loader.data_sets.geneva_stroke_outcome_dataset import GenevaStrokeOutcomeDataset

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
    def __init__(self, imaging_dataset_path, outcome_file_path, channels, outcome, preload_data, batch_size,
                 shuffle=True, validation_split=0.0, num_workers=1, augmentation=True):

        if augmentation:
            transforms = gsd_pCT_train_transform()
        else:
            transforms = None

        self.dataset = GenevaStrokeOutcomeDataset(imaging_dataset_path, outcome_file_path, channels, outcome, transform=transforms, preload_data=preload_data)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


