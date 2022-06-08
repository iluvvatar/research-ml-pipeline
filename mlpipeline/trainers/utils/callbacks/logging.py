import json

from .callback import Callback


class LoggingCallback(Callback):
    def __init__(self, print_: bool = False):
        self.print_ = print_

    def __call__(self, trainer):
        epoch = trainer.history.epoch
        start = trainer.history.start_timestamp[-1]
        stop = trainer.history.end_timestamp[-1]
        train_loss = trainer.history.train_loss[-1]
        val_loss = trainer.history.val_loss[-1]
        metrics = {metric: values[-1] for metric, values in trainer.history.metrics.items()}

        if self.print_:
            fmt = '%d.%m.%Y %H:%M:%S'
            print(f'Epoch {trainer.history.epoch} finished '
                  f'({start.strftime(fmt)} - {stop.strftime(fmt)}). '
                  f'Train loss {train_loss}. Val loss {val_loss}. Metrics {metrics}.')

        path = trainer.output_dir / 'log.jsonl'
        log = {'epoch': epoch,
               'start_timestamp': str(start),
               'end_timestamp': str(stop),
               'train_loss': train_loss,
               'val_loss': val_loss,
               'metrics': metrics}
        if not path.parent.exists():
            path.parent.mkdirs(parent=True)
        with open(path, 'a', encoding='utf-8') as f:
            print(json.dumps(log, ensure_ascii=False), file=f)
