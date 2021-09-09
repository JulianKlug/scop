import os
import shutil
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from keras_scope.test import test
from keras_scope.utils import ensure_dir, save_dataset
from keras_scope.train import train

def cross_validate():
    n_repeats = 1
    n_folds = 5

    label_file_path = '/mnt/data/hendrik/jk/scope_data/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx'
    imaging_dataset_path = '/mnt/data/hendrik/jk/scope_data/noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz'
    output_dir = '/home/hendrik/jk/output/keras_scope/cross_validation'

    # imaging_dataset_path = "/Users/jk1/stroke_datasets/dataset_files/perfusion_data_sets/noGT_datasets/train_noGT_pmaps_15-19_dataset_with_combined_mRS_90_days.npz"
    # label_file_path = "/Users/jk1/temp/scope_test/joined_anon_outcomes_2015_2016_2017_2018_df.xlsx"
    # output_dir = '/Users/jk1/temp/cv_scope_test'

    channels = [0, 1, 2, 3]
    outcome = "combined_mRS_0-2_90_days"
    desired_shape = (46, 46, 46)
    epochs = 400
    initial_learning_rate = 0.0001

    ensure_dir(output_dir)
    output_dir = os.path.join(output_dir, 'cv_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    ensure_dir(output_dir)

    # load data
    params = np.load(imaging_dataset_path, allow_pickle=True)['params']
    ids = np.load(imaging_dataset_path, allow_pickle=True)['ids']
    outcomes_df = pd.read_excel(label_file_path)
    labels = np.array([outcomes_df.loc[outcomes_df['anonymised_id'] == subj_id, outcome].iloc[0] for
                  subj_id in ids])
    raw_images = np.load(imaging_dataset_path, allow_pickle=True)['ct_inputs']
    raw_masks = np.load(imaging_dataset_path, allow_pickle=True)['brain_masks']
    all_indices = list(range(len(ids)))

    result_df = pd.DataFrame()

    # Start iteration of repeated k-fold cross-validation
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(output_dir, 'iteration_' + str(iteration))
        ensure_dir(iteration_dir)

        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = j)
        for train_indices, test_indices in kf.split(all_indices, labels):
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))
            ensure_dir(fold_dir)

            # save temporary dataset files for this fold
            temp_data_dir = tempfile.mkdtemp()
            temp_train_data_path = os.path.join(temp_data_dir, 'train_dataset.npz')
            temp_test_data_path = os.path.join(temp_data_dir, 'test_dataset.npz')
            save_dataset(raw_images[train_indices], raw_masks[train_indices], ids[train_indices], params, temp_train_data_path)
            save_dataset(raw_images[test_indices], raw_masks[test_indices], ids[test_indices], params, temp_test_data_path)

            # train
            _, model_path = train(label_file_path, temp_train_data_path, fold_dir, outcome, channels, desired_shape,
                                    initial_learning_rate, epochs)

            # test
            fold_result_dict = test(model_path, label_file_path, temp_test_data_path, outcome, channels, desired_shape)

            # store results
            fold_result_dict.update({'iteration': iteration, 'fold': fold, 'kfold_split_seed': j})
            fold_result_df = pd.DataFrame(fold_result_dict, index=[0])
            result_df = result_df.append(fold_result_df)

            # todo add all parameters to train and test

            shutil.rmtree(temp_data_dir)
            fold += 1
        iteration += 1

    print('Median test AUC', result_df['auc'].median())
    print('Median test accuracy', result_df['acc'].median())
    result_df.to_csv(os.path.join(output_dir, 'cv_test_results.csv'))


if __name__ == '__main__':
    cross_validate()
