import torch, os, optuna
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import pandas as pd


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, hyperopt_trial=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.hyperopt_trial = hyperopt_trial

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        if hyperopt_trial is not None:
            self.hyperopt_monitor = cfg_trainer['hyperopt_monitor']
            self.hyperopt_monitor_best = 0

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.experiment_dataframe_path = config.config['trainer']['experiment_dataframe_path']

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # parameter evaluation for hyperparameter tuning
            if self.hyperopt_trial is not None:
                self.hyperopt_trial.report(log[self.hyperopt_monitor], epoch)
                if log[self.hyperopt_monitor] >= self.hyperopt_monitor_best:
                    self.hyperopt_monitor_best = log[self.hyperopt_monitor]

                # Handle pruning based on the intermediate value.
                if self.hyperopt_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self._update_experiments_dataframe(epoch, result)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _update_experiments_dataframe(self, epoch, result_dict):

        experiment_dict = dict({
            'experiment_uid': self.config.uid,
            'architecture': type(self.model).__name__,
            'epoch': epoch,
            'optimizer': type(self.optimizer).__name__,
            'loss_function': self.config.config['loss'],
            'monitor': self.monitor,
            "imaging_dataset_path": self.config.config['data_loader']['args']['imaging_dataset_path'],
            "outcome_file_path": self.config.config['data_loader']['args']['outcome_file_path'],
            "checkpoint_dir": self.checkpoint_dir,
            "channels": [self.config.config['data_loader']['args']['channels']],
            "outcome": self.config.config['data_loader']['args']['outcome'],
            "augmentation": self.config.config['data_loader']['args']['augmentation'],
            "batch_size": self.config.config['data_loader']['args']['batch_size'],
        })

        experiment_dict.update(self.config.config['optimizer']['args'])
        experiment_dict.update(self.config.config['arch']['args'])
        experiment_dict.update(result_dict)

        current_experiment_df = pd.DataFrame(experiment_dict, index=[0])

        if os.path.exists(self.experiment_dataframe_path):
            all_experiments_df = pd.read_csv(self.experiment_dataframe_path)
            if not self.config.uid in all_experiments_df['experiment_uid'].values:
                all_experiments_df = pd.concat([all_experiments_df, current_experiment_df], axis=0, ignore_index=True)
            else:
                all_experiments_df.loc[
                    all_experiments_df['experiment_uid'] == self.config.uid] = current_experiment_df
            all_experiments_df.to_csv(self.experiment_dataframe_path, index=False)
        else:
            current_experiment_df.to_csv(self.experiment_dataframe_path, index=False)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
