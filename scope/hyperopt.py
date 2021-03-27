import argparse, os
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
from parse_config import ConfigParser
from trainer import Trainer
from pathlib import Path
from utils import prepare_device, ensure_dir, write_json

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


class ObjectiveConfigurator():
    def __init__(self, config):
        self.config = config

    def objective(self, trial):
        self.config.uid = uuid.uuid4().hex
        self.config._log_dir = Path(os.path.join(self.config.log_dir.parent, self.config.uid))
        self.config._save_dir = Path(os.path.join(self.config.save_dir, self.config.uid))
        ensure_dir(self.config.log_dir)
        ensure_dir(self.config.save_dir)

        # overwrite parameters chosen by hyperoptimization
        optimizer_name = trial.suggest_categorical("optimizer", self.config.config['optimizer']['type'])
        self.config.config['optimizer']['type'] = optimizer_name
        lr = trial.suggest_float("lr", self.config.config['optimizer']["args"]["lr"][0], self.config.config['optimizer']["args"]["lr"][1], log=True)
        self.config.config['optimizer']["args"]["lr"] = lr

        lr_scheduler_name = trial.suggest_categorical("lr_scheduler", self.config.config['lr_scheduler']['type'])
        self.config.config['lr_scheduler']['type'] = lr_scheduler_name
        if lr_scheduler_name == "StepLR":
            self.config.config['lr_scheduler']['args']["step_size"] = 50

        drop_connect_rate = trial.suggest_float("drop_connect_rate", self.config.config['arch']['args']["drop_connect_rate"][0],
                                                self.config.config['arch']['args']["drop_connect_rate"][1], step=0.1)
        self.config.config['arch']['args']["drop_connect_rate"] = drop_connect_rate

        batch_size = trial.suggest_categorical("batch_size", self.config.config['data_loader']['args']['batch_size'])
        self.config.config['data_loader']['args']['batch_size'] = batch_size

        # save new config
        write_json(self.config.config, self.config.save_dir / 'config.json')

        # Now initiating all objects from the new config
        logger = self.config.get_logger('hyperopt')

        # setup data_loader instances
        data_loader = self.config.init_obj('data_loader', module_data)
        valid_data_loader = data_loader.split_validation()

        # build model architecture, then print to console
        model = self.config.init_obj('arch', module_arch)
        logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, self.config['loss'])
        metrics = [getattr(module_metric, met) for met in self.config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self.config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=self.config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          hyperopt_trial=trial)

        trainer.train()


def hyperopt(config):
    # todo enable to load from saved prior study
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    configured_objective = ObjectiveConfigurator(config)

    study = optuna.create_study(study_name=config.config['name'] + '_' + run_id, direction="maximize",
                                sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    joblib.dump(study, os.path.join(config.save_dir, "hyperopt_study.pkl"))
    study.optimize(configured_objective.objective, n_trials=100, timeout=12000)
    joblib.dump(study, os.path.join(config.save_dir, "hyperopt_study.pkl"))

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
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    options = []
    config = ConfigParser.from_args(args, options)
    hyperopt(config)
