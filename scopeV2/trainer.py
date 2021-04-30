import hashlib
import shutil
import sys
import torch
import os
import argparse
import datetime
import torch.nn as nn
from matplotlib import pyplot
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import scopeV2.datasets as datasets
from scopeV2.models.luna_model import LunaModel
from scopeV2.utils.utils import enumerateWithEstimate, write_json
from scopeV2.utils.log_config import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1  # index of predictions in metrics
METRICS_PRED_P_NDX = 2    # index of prediction probability in metrics
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4


class Trainer:
    def __init__(self, config: dict):

        self.config = config

        self.time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.config.augmented or self.config.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.config.augmented or self.config.augment_offset:
            self.augmentation_dict['offset'] = 0.4
        if self.config.augmented or self.config.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.config.augmented or self.config.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.config.augmented or self.config.augment_noise:
            self.augmentation_dict['noise'] = 0.005

        self.validation_cadence = 5
        self.validation_size = self.config.validation_size

        self.dataset_class = getattr(datasets, self.config.dataset)
        self.monitoring_metric = self.config.monitoring_metric

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.batch_size = self.config.batch_size
        if self.use_cuda:
            self.batch_size *= torch.cuda.device_count()

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel(in_channels=len(self.config.channels))
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def initOptimizer(self):
        return Adam(self.model.parameters())
        # return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = self.dataset_class(
            self.config.data_path,
            self.config.label_path,
            outcome=self.config.outcome,
            channels=self.config.channels,
            validation_size=self.validation_size,
            is_validation=False,
            neg_to_pos_ratio=int(self.config.balanced),
            augmentation_dict=self.augmentation_dict,
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = self.dataset_class(
            self.config.data_path,
            self.config.label_path,
            outcome=self.config.outcome,
            channels=self.config.channels,
            validation_size=self.validation_size,
            is_validation=True
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.config))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        for epoch_ndx in range(1, self.config.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.config.epochs,
                len(train_dl),
                len(val_dl),
                self.config.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                # if validation is wanted
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('classifier', epoch_ndx, score == best_score)

            if epoch_ndx == 1:
                self.saveConfig()

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

        return best_score

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()

        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, id = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g,
            label_g[:, 1]
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False,
                                   out=None)

        # store label, prediction, probability and loss as metrics
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join(self.config.output_path, 'runs', self.config.experiment_name, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.config.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.config.comment)

    def logMetrics(
            self,
            epoch_ndx: int,
            mode_str: str,
            metrics_t,
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
                                      / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0, steps=100)
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:] - fpr[:-1]
        tp_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
             + "{auc:.4f} auc"
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)
        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        bins = np.linspace(0, 1)

        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        score = metrics_dict[self.monitoring_metric]

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            self.config.output_path,
            'models',
            self.config.experiment_name,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.config.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                self.config.output_path, 'models',
                self.config.experiment_name,
                f'{type_str}_{self.time_str}_{self.config.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def saveConfig(self):
        file_path = os.path.join(
            self.config.output_path,
            'models',
            self.config.experiment_name,
            f'{self.time_str}_{self.config.comment}_config.json'
        )
        write_json(self.config.__dict__, file_path)


if __name__ == '__main__':
    Trainer().main()
