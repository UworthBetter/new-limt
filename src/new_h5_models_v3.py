"""qlib v0.9.6 fit copies, with full-window IC diagnostics and RNG isolation.
The original loss loaders, train_epoch, test_epoch, optimizer, patience, save
and best-weight reload are retained. Only IC configurations replace val_score.
"""
# Fit implementations adapted from Microsoft Qlib v0.9.6 (MIT License).
# Copyright (c) Microsoft Corporation.
import copy
import numpy as np
import torch
from qlib.data.dataset import DatasetH
from torch.utils.data import DataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.weight import Reweighter
from qlib.utils import get_or_create_path
from qlib.contrib.model.pytorch_lstm_ts import LSTM, ConcatDataset
from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
from new_h5_common_v3 import isolated_rng


class DiagnosticMixin:
    def diagnostic(self, dataset, step, train_loss, val_loss, val_score):
        ic = None
        if self.diagnostics_enabled or self.early_stop_metric == 'valid_ic5':
            with isolated_rng():
                ic = self.ic_callback(self, dataset)
            if ic is None or not np.isfinite(ic):
                raise ValueError('undefined whole-window validation IC')
        self.per_epoch.append(dict(epoch=step, train_loss=float(train_loss),
            valid_loss=float(val_loss), valid_ic5=ic))
        return ic if self.early_stop_metric == 'valid_ic5' else val_score


class GuardedLSTM(DiagnosticMixin, LSTM):
    early_stop_metric = "valid_loss"
    diagnostics_enabled = True

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        self.per_epoch = []
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            val_score = self.diagnostic(dataset, step, train_loss, val_loss, val_score)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.LSTM_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.LSTM_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        self.best_epoch = best_epoch
        if self.use_gpu:
            torch.cuda.empty_cache()


class GuardedTransformer(DiagnosticMixin, TransformerModel):
    early_stop_metric = "valid_loss"
    diagnostics_enabled = True

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        self.per_epoch = []
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            val_score = self.diagnostic(dataset, step, train_loss, val_loss, val_score)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        self.best_epoch = best_epoch
        if self.use_gpu:
            torch.cuda.empty_cache()


