import argparse, os
import copy
import shutil
from datetime import datetime
import uuid

import joblib
import optuna
from optuna.trial import TrialState
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from archives.scope.parse_config import ConfigParser
from trainer import Trainer
from pathlib import Path
from archives.scope.utils import prepare_device, ensure_dir, write_json

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def objective_configurator(config):
    def objective(trial):
        objective_config = copy.deepcopy(config)
        objective_config.uid = uuid.uuid4().hex
        objective_config._log_dir = Path(os.path.join(objective_config.log_dir.parent, objective_config.uid))
        objective_config._save_dir = Path(os.path.join(objective_config.save_dir, objective_config.uid))
        ensure_dir(objective_config.log_dir)
        ensure_dir(objective_config.save_dir)

        # overwrite parameters chosen by hyperoptimization
        optimizer_name = trial.suggest_categorical("optimizer", objective_config.config['optimizer']['type'])
        objective_config.config['optimizer']['type'] = optimizer_name
        lr = trial.suggest_float("lr", objective_config.config['optimizer']["args"]["lr"][0], objective_config.config['optimizer']["args"]["lr"][1], log=True)
        objective_config.config['optimizer']["args"]["lr"] = lr

        lr_scheduler_name = trial.suggest_categorical("lr_scheduler", objective_config.config['lr_scheduler']['type'])
        objective_config.config['lr_scheduler']['type'] = lr_scheduler_name
        if lr_scheduler_name == "StepLR":
            objective_config.config['lr_scheduler']['args']["step_size"] = 50

        drop_connect_rate = trial.suggest_float("drop_connect_rate", objective_config.config['arch']['args']["drop_connect_rate"][0],
                                                objective_config.config['arch']['args']["drop_connect_rate"][1], step=0.1)
        objective_config.config['arch']['args']["drop_connect_rate"] = drop_connect_rate

        batch_size = trial.suggest_categorical("batch_size", objective_config.config['data_loader']['args']['batch_size'])
        objective_config.config['data_loader']['args']['batch_size'] = batch_size

        # save new config
        write_json(objective_config.config, objective_config.save_dir / 'config.json')

        # Now initiating all objects from the new config
        logger = objective_config.get_logger('hyperopt')

        # setup data_loader instances
        data_loader = objective_config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = objective_config.init_obj('arch', module_arch)
        logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(objective_config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, objective_config['loss'])
        metrics = [getattr(module_metric, met) for met in objective_config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = objective_config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = objective_config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=objective_config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          hyperopt_trial=trial)

        trainer.train()

        return trainer.hyperopt_monitor_best
    return objective


class StudySaverCallback:
    def __init__(self, save_path: str):
        self.save_path = save_path

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        joblib.dump(study, self.save_path)


def hyperopt(config, old_study_path=None):

    if old_study_path is None:
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        study = optuna.create_study(study_name=config.config['name'] + '_' + run_id, direction="maximize",
                                    sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
        study_path = os.path.join(config.save_dir, "hyperopt_study.pkl")
        joblib.dump(study, study_path)
    else:
        study = joblib.load(old_study_path)
        study_path = old_study_path

        shutil.rmtree(config.save_dir)
        config._save_dir = os.path.dirname(study_path)

    study_saver_cb = StudySaverCallback(study_path)
    study.optimize(objective_configurator(config), callbacks=[study_saver_cb], timeout=162000)  # 45h
    joblib.dump(study, study_path)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--study_path', default=None, type=str,
                      help='Path to already existent hyperopt study (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    options = []
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    hyperopt(config, args.study_path)
