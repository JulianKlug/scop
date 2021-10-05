import os
from archives.scopeV2.datasets import LeftRightDataset
import pytest


@pytest.mark.dataset
def test_dataset_creation():
    ds = LeftRightDataset(os.environ.get('DATA_PATH'), os.environ.get('LABEL_PATH'), validation_size=0,
                          neg_to_pos_ratio=1)


@pytest.mark.dataset
def test_dataset_neg_to_pos_ratio():
    ds = LeftRightDataset(os.environ.get('DATA_PATH'), os.environ.get('LABEL_PATH'), validation_size=0,
                          neg_to_pos_ratio=1)
    labels = [ds[idx][1][0] for idx in range(50)]
    assert sum(labels) / 50 == 0.5


if __name__ == '__main__':
    test_dataset_creation()
