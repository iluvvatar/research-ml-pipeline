from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any, Union, Callable
from pathlib import Path
from datetime import datetime, timezone

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..models import Model
from ..utils import PathLike
from .utils import History, Callback, Metric, Loss, LRScheduler

from tqdm import trange, tqdm


class Trainer(ABC):
    def __init__(self, *,
                 model: Model,
                 optimizer: Optimizer,
                 loss_fn: Loss,
                 lr_scheduler: LRScheduler,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 val_postprocess_batch_fn: Callable[
                     [Dict[str, List[Any]]], Dict[str, List[Any]]
                 ] = None,
                 callbacks: List[Callback],
                 metrics: List[Metric],
                 key_metric_name: str,
                 device: str,
                 output_dir: PathLike,
                 verbose: bool = True,
                 max_grad_norm: float = 1):
        assert device in ['cpu', 'cuda']
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm

        self.train_loader = train_loader
        self.val_loader = val_loader
        if val_postprocess_batch_fn is None:
            self.val_postprocess_batch_fn = lambda batch: batch
        else:
            self.val_postprocess_batch_fn = val_postprocess_batch_fn

        self.callbacks = callbacks
        self.metrics = metrics
        self.key_metric_name = key_metric_name
        self.history = History.create(metrics)

        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.verbose = verbose

    # Training
    # =========================================================================
    def train(self, epochs: int):
        start_epoch = self.history.epoch + 1
        train_loss, val_loss, metrics = self.init_history()
        try:
            desc_pattern = 'Epoch {}. Train loss {}. Val loss {}. Metrics {}.'
            pbar = trange(start_epoch, epochs + 1, disable=not self.verbose)
            for epoch in pbar:
                desc = desc_pattern.format(epoch, train_loss, val_loss, metrics)
                pbar.set_description(desc)

                # Training
                epoch_start = datetime.now(timezone.utc)
                train_loss = self.train_epoch()
                epoch_finish = datetime.now(timezone.utc)

                # Validation
                val_loss, metrics = self.eval_epoch()

                # Update state
                self.history.add(start_timestamp=epoch_start,
                                 end_timestamp=epoch_finish,
                                 train_loss=train_loss,
                                 val_loss=val_loss,
                                 metrics=metrics)
                self.scheduler.step(self)
                for callback in self.callbacks:
                    callback(trainer=self)
        except StopIteration:
            print('Training was stopped')

    def train_epoch(self) -> float:
        self.model.train()
        loss = 0
        for batch in tqdm(self.train_loader, desc=f'Training', disable=not self.verbose):
            loss += self.train_step(batch)
        return loss

    def eval_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        for metric in self.metrics:
            metric.clear()
        metrics = {}
        loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Evaluation', disable=not self.verbose):
                loss += self.eval_step(batch)
            for metric in self.metrics:
                computed = metric.compute(self)
                for name in metric.names:
                    # костыль для порядка в логе
                    metrics[name] = computed[name]
        return loss, metrics

    @abstractmethod
    def train_step(self, batch: Any) -> float:
        """
        Zero gradients, handle Batch, do optimizer step and return loss.
        Used in training.

        self.optimizer.zero_grad()
        pred_y = ...
        loss = self.loss_fn(...)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        """
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, batch: Dict[str, Any]) -> float:
        """
        Compute loss and add update metrics with predictions
        via Metric.add_batch().
        Used in training.

        batch['pred'] = ...
        loss = self.loss_fn(...)
        for metric in metrics:
            metric.add_batch(batch)
        return loss.item()
        """
        raise NotImplementedError

    def init_history(self):
        if self.history.is_empty():
            train_loss = None
            self.model.eval()
            epoch_start = datetime.now(timezone.utc)
            val_loss, metrics = self.eval_epoch()
            epoch_finish = datetime.now(timezone.utc)
            self.history.add(start_timestamp=epoch_start,
                             end_timestamp=epoch_finish,
                             train_loss=None,
                             val_loss=val_loss,
                             metrics=metrics)
            for callback in self.callbacks:
                callback(trainer=self)
        else:
            train_loss = self.history.train_loss[-1]
            val_loss = self.history.val_loss[-1]
            metrics = {k: v[-1] for k, v in self.history.metrics.items()}

        return train_loss, val_loss, metrics

    # Save / Load
    # =========================================================================
    # saving moved to CheckpointCallback
    def load_checkpoint(self, path: PathLike,
                        load_optimizer: bool = True,
                        load_loss_fn: bool = False,
                        load_scheduler: bool = True):
        path = Path(path)
        self.model.load_state_dict(
            torch.load(path / 'model.pt', map_location=self.device)
        )
        self.history = History.load(path / 'history.json')
        assert set(name for m in self.metrics for name in m.names) \
               == set(self.history.metrics.keys())
        if load_optimizer:
            self.optimizer.load_state_dict(
                torch.load(path / 'optim.pt', map_location=self.device)
            )
        if load_scheduler:
            self.scheduler.load_state_dict(
                torch.load(path / 'scheduler.pt', map_location=self.device)
            )
        if load_loss_fn:
            self.loss_fn = Loss.load(path / 'loss.json').to(self.device)
