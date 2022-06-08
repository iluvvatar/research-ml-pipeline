import numpy as np
import copy
import torch
from pathlib import Path

from .callback import Callback


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir_path, cooldown=1, score_mode='max',
                 save_optimizer: bool = True):
        """
        Parameters
        ----------
        checkpoint_dir_path: PathLike
        cooldown : int
            Minimal epochs between making checkpoint.
        score_mode : str
        save_optimizer : bool
        """
        assert score_mode in ['max', 'min']
        self.checkpoint_dir_path = Path(checkpoint_dir_path)
        self.score_mode = score_mode
        self.cooldown = cooldown
        self.save_optimizer = save_optimizer
        self.best_model_state_dict = None
        self.best_epoch = 0
        if score_mode == 'max':
            self.best_score = -np.inf
        else:
            self.best_score = np.inf
        self.saved_epochs = set()

    def __call__(self, trainer):
        epoch = trainer.history.epoch
        score = trainer.history.metrics[trainer.key_metric_name][-1]
        if self.score_mode == 'max' and score > self.best_score \
                or self.score_mode == 'min' and score < self.best_score:
            if self.save_optimizer:
                self.best_optimizer_state_dict = copy.deepcopy(trainer.optimizer.state_dict())
            else:
                self.best_optimizer_state_dict = None
            self.best_model_state_dict = copy.deepcopy(trainer.model.state_dict())
            self.best_scheduler_state_dict = copy.deepcopy(trainer.scheduler.state_dict())
            self.best_loss = copy.deepcopy(trainer.loss_fn)
            self.best_history = copy.deepcopy(trainer.history)
            self.best_score = score
            self.best_epoch = epoch
        if epoch > 0 and epoch % self.cooldown == 0:
            # make checkpoint
            if self.best_epoch not in self.saved_epochs:
                self.saved_epochs.add(self.best_epoch)
                score_str = f'{trainer.key_metric_name}={self.best_score}'
                checkpoint_name = f'{trainer.model.name}_epoch={self.best_epoch}_{score_str}'
                self.save_checkpoint(checkpoint_name,
                                     self.best_model_state_dict,
                                     self.best_optimizer_state_dict,
                                     self.best_scheduler_state_dict,
                                     self.best_loss,
                                     self.best_history)
            if epoch != self.best_epoch:
                self.saved_epochs.add(epoch)
                score_str = f'{trainer.key_metric_name}={score}'
                checkpoint_name = f'{trainer.model.name}_epoch={epoch}_{score_str}'
                if self.save_optimizer:
                    optim = trainer.optimizer.state_dict()
                else:
                    optim = None
                self.save_checkpoint(checkpoint_name,
                                     trainer.model.state_dict(),
                                     optim,
                                     trainer.scheduler.state_dict(),
                                     trainer.loss_fn,
                                     trainer.history)

    def save_checkpoint(self, checkpoint_name,
                        model_state_dict,
                        optimizer_state_dict,
                        scheduler_state_dict,
                        loss, history):
        """
        Parameters
        ----------
        checkpoint_name : str
        model_state_dict : dict
        optimizer_state_dict : dict
        scheduler_state_dict : dict
        loss : Loss
        history : History
        """
        path = self.checkpoint_dir_path / checkpoint_name
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_state_dict, path / 'model.pt')
        if self.save_optimizer:
            torch.save(optimizer_state_dict, path / 'optim.pt')
        torch.save(scheduler_state_dict, path / 'scheduler.pt')
        loss.save(path / 'loss.json')
        history.save(path / 'history.json')
