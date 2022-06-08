from .callback import Callback
from ..losses.loss import LossList


class LossListAlphaStepCallback(Callback):
    def __init__(self, factor, alpha_min=0, alpha_max=1):
        """
        Parameters
        ----------
        factor : float
        alpha_min : float
        alpha_max : float
        """
        assert 0 <= alpha_min and alpha_max <= 1
        self.factor = factor
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def __call__(self, trainer):
        if trainer.history.epoch == 0:
            return
        loss = trainer.loss_fn
        assert isinstance(loss, LossList)
        assert len(loss) == 2
        alpha = loss.ratios[0]
        if self.factor > 1:
            if alpha < 0.5:
                alpha = min(0.5, alpha * self.factor)
            else:
                alpha = min(self.alpha_max, 1 - (1 - alpha) / self.factor)
        else:
            if alpha <= 0.5:
                alpha = max(self.alpha_min, alpha * self.factor)
            else:
                alpha = max(0.5, 1 - (1 - alpha) / self.factor)
        loss.ratios.data[0] = alpha
        loss.ratios.data[1] = 1 - alpha
