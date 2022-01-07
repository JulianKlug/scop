import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import uuid
from norby.utils import maybe_norby
from sklearn.model_selection import StratifiedKFold
import os
import tempfile

from scope.parse_config import parse_config
from scope.test import test
from scope.train import train
from scope.utils.utils import ensure_dir, save_dataset
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback


def tunable_train(config):
    label_file_path = config['label_file_path']
    imaging_dataset_path = config['imaging_dataset_path']
    output_dir = config['output_dir']
    channels = config['channels']
    outcome = config['outcome']
    model_type = config['model_type']
    model_input_shape = config['model_input_shape']
    epochs = config['epochs']

    # load data
    params = np.load(imaging_dataset_path, allow_pickle=True)['params']
    ids = np.load(imaging_dataset_path, allow_pickle=True)['ids']
    outcomes_df = pd.read_excel(label_file_path)
    labels = np.array([outcomes_df.loc[outcomes_df[config['id_variable']] == subj_id, outcome].iloc[0] for
                       subj_id in ids])
    raw_images = np.load(imaging_dataset_path, allow_pickle=True)['ct_inputs']
    raw_masks = np.load(imaging_dataset_path, allow_pickle=True)['brain_masks']
    all_indices = list(range(len(ids)))


    np.random.seed(42)
    j = np.random.randint(0, high=10000, size=1)[0]
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=j)
    train_indices, test_indices = list(kf.split(all_indices, labels))[0]


    # save temporary dataset files for this fold
    temp_data_dir = tempfile.mkdtemp()
    temp_train_data_path = os.path.join(temp_data_dir, 'train_dataset.npz')
    temp_test_data_path = os.path.join(temp_data_dir, 'test_dataset.npz')
    save_dataset(raw_images[train_indices], raw_masks[train_indices], ids[train_indices], params,
                 temp_train_data_path)
    save_dataset(raw_images[test_indices], raw_masks[test_indices], ids[test_indices], params,
                 temp_test_data_path)

    hyperopt_sample_id = str(uuid.uuid4().hex)
    hyperopt_run_dir = os.path.join(output_dir, config["hyperopt_id"])
    hyperopt_sample_dir = os.path.join(hyperopt_run_dir, hyperopt_sample_id)
    _, model_paths, saved_train_epoch, best_val_score, best_val_score_plateau = train(
        label_file_path,
         temp_train_data_path, hyperopt_run_dir,
         outcome,
         channels, model_type,
         model_input_shape,
         config['initial_learning_rate'],
         epochs=epochs,
         target_metric=config['target_metric'],
         id_variable=config['id_variable'],
         split_ratio=config['validation_size'],
         batch_size=config['batch_size'],
         early_stopping_patience=config['early_stopping_patience'],
         use_augmentation=config['use_augmentation'],
         augmentation_magnitude=config['augmentation_magnitude'],
         continuous_outcome=config['continuous_outcome'],
         lr_decay_steps=config['lr_decay_steps'],
         weight_decay_coefficient=config['weight_decay_coefficient'],
         logdir=hyperopt_sample_dir,
         extra_callbacks=[TuneReportCallback({
                                "auc": "val_auc"
                                })])


    # test
    fold_result_dict, _ = test(model_paths, model_type, label_file_path,
                                                      temp_test_data_path, outcome,
                                                      channels, model_input_shape,
                                                      id_variable=config['id_variable'],
                                                      single_subject_predictions=True,
                                                      continuous_outcome=config['continuous_outcome'],
                                                      weight_decay_coefficient=config['weight_decay_coefficient'])

    # store results
    fold_result_dict.update({'hyperopt_id': config["hyperopt_id"], "hyperopt_sample_id": hyperopt_sample_id, 'kfold_split_seed': j,
                             'train_epoch': saved_train_epoch,
                             'validation_auc': best_val_score, 'validation_plateau_auc': best_val_score_plateau})

    hyperopt_run_result_df = pd.DataFrame(fold_result_dict, index=[0])
    hyperopt_result_path = os.path.join(output_dir, 'hyperopt_test_results.csv')
    if os.path.exists(hyperopt_result_path):
        hyperopt_run_result_df.to_csv(hyperopt_result_path, mode='a', header=False)
    else:
        hyperopt_run_result_df.to_csv(hyperopt_result_path)

    # clean up
    shutil.rmtree(temp_data_dir)
    # shutil.rmtree(model_paths)


def tune_onset_scope(config, num_training_iterations=300):
    # ip_head and redis_passwords are set by ray cluster shell scripts
    print(os.environ["ip_head"], os.environ["redis_password"])
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
             _redis_password=os.environ["redis_password"])
    sched = ASHAScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20)

    config = vars(config)
    config.update({
        "initial_learning_rate": tune.uniform(0.001, 0.1),
        "augmentation_magnitude": tune.uniform(0.25, 10),
        "lr_decay_steps": tune.choice([25, 50, 500, 50000, 100000]),
        "weight_decay_coefficient": tune.uniform(0.0, 0.1),
        "batch_size": tune.choice([2, 10, 32, 64])
    })

    config["hyperopt_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperopt_run_dir = os.path.join(config["output_dir"], config["hyperopt_id"])
    ensure_dir(hyperopt_run_dir)

    analysis = tune.run(
        tunable_train,
        name="onset_scope_hyperoptimization",
        scheduler=sched,
        metric="auc",
        mode="max",
        stop={
            "auc": 0.9,
            "training_iteration": num_training_iterations
        },
        num_samples=2,
        resources_per_trial={
            "cpu": 10,
            "gpu": 1
        },
        config=config)
    print("Best hyperparameters found were: ", analysis.best_config)
    # save best config
    best_config_path = os.path.join(config['output_dir'], 'best_config.json')
    with open(best_config_path, 'w') as f:
        json.dump(analysis.best_config, f)


if __name__ == '__main__':
    config = parse_config()
    with maybe_norby(config.norby, f'Starting Hyperoptimization {config.experiment_id}{config.comment}.',
                     f'Hyperoptimization finished {config.experiment_id}.', whichbot='scope'):
        tune_onset_scope(config, num_training_iterations=3)
