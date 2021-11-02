import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from modun.file_io import dict2json
from norby.utils import maybe_norby

from parse_config import parse_config
from scope.test import test
from scope.utils.utils import ensure_dir, save_dataset
from scope.utils.data_splitting import SortedStratifiedKFold
from scope.train import train


def cross_validate(config: dict):
    n_repeats = config.cv_n_repeats
    n_folds = config.cv_n_folds

    label_file_path = config.label_file_path
    imaging_dataset_path = config.imaging_dataset_path
    output_dir = config.output_dir

    channels = config.channels
    outcome = config.outcome
    model_type = config.model_type
    model_input_shape = config.model_input_shape
    epochs = config.epochs
    initial_learning_rate = config.initial_learning_rate

    ensure_dir(output_dir)
    output_dir = os.path.join(output_dir, 'cv_' + config.experiment_id + config.comment)
    ensure_dir(output_dir)
    dict2json(os.path.join(output_dir, 'config.json'), vars(config))

    # load data
    params = np.load(imaging_dataset_path, allow_pickle=True)['params']
    ids = np.load(imaging_dataset_path, allow_pickle=True)['ids']
    outcomes_df = pd.read_excel(label_file_path)
    labels = np.array([outcomes_df.loc[outcomes_df[config.id_variable] == subj_id, outcome].iloc[0] for
                       subj_id in ids])
    raw_images = np.load(imaging_dataset_path, allow_pickle=True)['ct_inputs']
    raw_masks = np.load(imaging_dataset_path, allow_pickle=True)['brain_masks']
    all_indices = list(range(len(ids)))

    result_df = pd.DataFrame()
    test_prediction_df = pd.DataFrame()

    # Start iteration of repeated k-fold cross-validation
    iteration = 0
    for j in np.random.randint(0, high=10000, size=n_repeats):
        iteration_dir = os.path.join(output_dir, 'iteration_' + str(iteration))
        ensure_dir(iteration_dir)

        print('Crossvalidation: Creating iteration ' + str(iteration) + ' of a total of ' + str(n_repeats))

        fold = 0
        if config.continuous_outcome:
            kf = SortedStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=j)
        else:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=j)
        for train_indices, test_indices in kf.split(all_indices, labels):
            fold_dir = os.path.join(iteration_dir, 'fold_' + str(fold))
            ensure_dir(fold_dir)

            # save temporary dataset files for this fold
            temp_data_dir = tempfile.mkdtemp()
            temp_train_data_path = os.path.join(temp_data_dir, 'train_dataset.npz')
            temp_test_data_path = os.path.join(temp_data_dir, 'test_dataset.npz')
            save_dataset(raw_images[train_indices], raw_masks[train_indices], ids[train_indices], params,
                         temp_train_data_path)
            save_dataset(raw_images[test_indices], raw_masks[test_indices], ids[test_indices], params,
                         temp_test_data_path)

            model_paths = []
            best_val_score_plateaus = []
            # Multiple train rounds can be used to select best model based on validation scope or to construct ensemble
            for train_round_i in range(config.max_train_rounds):
                # train
                _, model_path, saved_train_epoch, best_val_score_plateau = train(label_file_path, temp_train_data_path, fold_dir, outcome,
                                                              channels, model_type, model_input_shape,
                                                              initial_learning_rate, epochs=epochs,
                                                              target_metric=config.target_metric,
                                                              id_variable=config.id_variable,
                                                              split_ratio=config.validation_size,
                                                              batch_size=config.batch_size,
                                                              early_stopping_patience=config.early_stopping_patience,
                                                              use_augmentation=config.use_augmentation,
                                                              augmentation_magnitude=config.augmentation_magnitude,
                                                              continuous_outcome=config.continuous_outcome,
                                                              lr_decay_steps=config.lr_decay_steps,
                                                              weight_decay_coefficient=config.weight_decay_coefficient)

                model_paths.append(model_path)
                best_val_score_plateaus.append(best_val_score_plateau)
                if best_val_score_plateau > config.min_val_score:
                    break

            # if not using an ensemble, use model with best validation score
            if not config.use_ensemble or len(model_paths) == 1:
                model_paths = model_paths[np.argmax(best_val_score_plateaus)]

            # test
            fold_result_dict, subject_prediction_label = test(model_paths, model_type, label_file_path,
                                                              temp_test_data_path, outcome,
                                                              channels, model_input_shape,
                                                              id_variable=config.id_variable,
                                                              single_subject_predictions=True,
                                                              continuous_outcome=config.continuous_outcome,
                                                              weight_decay_coefficient=config.weight_decay_coefficient)

            # store results
            fold_result_dict.update({'iteration': iteration, 'fold': fold, 'kfold_split_seed': j,
                                     'train_epoch':saved_train_epoch, 'validation_plateau_auc': best_val_score_plateau})
            subject_prediction_label = np.vstack(
                [subject_prediction_label, np.repeat(iteration, subject_prediction_label.shape[1]),
                 np.repeat(j, subject_prediction_label.shape[1])])
            fold_result_df = pd.DataFrame(fold_result_dict, index=[0])
            result_df = result_df.append(fold_result_df)

            fold_test_prediction_df = pd.DataFrame(subject_prediction_label.T,
                                                   columns=['id', 'test_prediction', 'test_label',
                                                            'cross_validation_iteration',
                                                            'cross_validation_kfold_split_seed'])
            test_prediction_df = test_prediction_df.append(fold_test_prediction_df, ignore_index=True)
            result_df.to_csv(os.path.join(output_dir, 'cv_test_results.csv'))
            test_prediction_df.to_csv(os.path.join(output_dir, 'cv_test_predictions.csv'))

            shutil.rmtree(temp_data_dir)
            fold += 1
        iteration += 1

    output_statement = f'Median test AUC: {result_df["auc"].median()} \n' \
                       f'Median test accuracy: {result_df["acc"].median()}'

    print(output_statement)
    return output_statement


if __name__ == '__main__':
    config = parse_config()
    with maybe_norby(config.norby, f'Starting Experiment {config.experiment_id}{config.comment}.',
                     f'Experiment finished {config.experiment_id}.', whichbot='scope'):
        cross_validate(config)
