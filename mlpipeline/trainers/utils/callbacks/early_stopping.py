from .callback import Callback


class EarlyStoppingException(Exception):
    def __init__(self, epoch, metric, prev_metrics):
        """
        Parameters
        ----------
        epoch : int
        metric : float
        prev_metrics : List[float]
        """
        super().__init__(f'Training was stopped as {epoch} epoch, '
                         f'because target metric stopped improving:\n'
                         f'metric value = {metric}, '
                         f'previous metric values = {prev_metrics}.')


class EarlyStoppingCallback(Callback):
    def __init__(self, *,
                 mode='max',
                 patience: int = 1,
                 threshold_mode='abs',
                 threshold: float = 0):
        """
        Parameters
        ----------
        mode : str
        patience : int
        threshold_mode : str
        threshold : float
        """
        assert patience > 0
        assert mode in ['min', 'max']
        assert threshold_mode in ['abs', 'rel']
        self.mode = mode
        self.patience = patience
        self.threshold_mode = threshold_mode
        self.threshold = threshold

    def __call__(self, trainer):
        epoch = trainer.history.epoch
        if epoch <= self.patience + 1:
            return
        history = trainer.history.metrics[trainer.key_metric_name]
        cur_value = history[-1]
        prev_values = history[-1 - self.patience:-1]
        if self.mode == 'max':
            if self.threshold_mode == 'abs':
                stop = all(cur_value < value + self.threshold for value in prev_values)
            else:
                stop = all(cur_value < value * (1 + self.threshold) for value in prev_values)
        else:
            if self.threshold_mode == 'abs':
                stop = all(cur_value > value - self.threshold for value in prev_values)
            else:
                stop = all(cur_value > value * (1 - self.threshold) for value in prev_values)
        if stop:
            raise EarlyStoppingException(epoch, cur_value, prev_values)
