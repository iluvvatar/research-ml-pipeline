from .callback import Callback


class FreezeEmbeddingsCallback(Callback):
    def __init__(self, epochs, is_frozen=False):
        """
        Parameters
        ----------
        epochs : int
        is_frozen : bool
        """
        self.epochs = epochs
        self.is_frozen = is_frozen

    def __call__(self, trainer):
        if len(trainer.history) <= self.epochs and not self.is_frozen:
            trainer.model.freeze_embeddings(requires_grad=False)
            self.is_frozen = True
        if len(trainer.history) > self.epochs and self.is_frozen:
            trainer.model.freeze_embeddings(requires_grad=True)
            self.is_frozen = False
