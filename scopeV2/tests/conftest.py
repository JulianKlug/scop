import os
import shutil
import tempfile
import urllib

import pytest


@pytest.fixture(scope="session", autouse=True)
def tests_setup_and_teardown():
    # Will be executed before the first test
    old_environ = dict(os.environ)
    print('Initialising environment')
    dirpath = tempfile.mkdtemp()

    DATA_PATH = os.path.join(dirpath, 'data.npy')
    LABEL_PATH = os.path.join(dirpath, 'labels.csv')
    OUTPUT_PATH = os.path.join(dirpath, 'output')

    test_data_url = 'https://github.com/JulianKlug/SimpleVoxel-3D/raw/main/leftright/data.npy'
    test_label_url = 'https://github.com/JulianKlug/SimpleVoxel-3D/raw/main/leftright/labels.csv'
    urllib.request.urlretrieve(test_data_url, DATA_PATH)
    urllib.request.urlretrieve(test_label_url, LABEL_PATH)

    os.environ.update({'DATA_PATH':DATA_PATH})
    os.environ.update({'LABEL_PATH':LABEL_PATH})
    os.environ.update({'OUTPUT_PATH':OUTPUT_PATH})



    yield
    # Will be executed after the last test
    shutil.rmtree(dirpath)
    os.environ.clear()
    os.environ.update(old_environ)