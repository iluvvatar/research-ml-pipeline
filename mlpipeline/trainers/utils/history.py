from typing import Iterable, List, Dict
from datetime import datetime, timezone
from pathlib import Path
import json
from dataclasses import dataclass

from .metrics import Metric
from ...utils import PathLike


@dataclass
class History:
    start_timestamp: List[datetime]
    end_timestamp: List[datetime]
    train_loss: List[float]
    val_loss: List[float]
    metrics: Dict[str, List[float]]

    @classmethod
    def create(cls, metrics: Iterable[Metric]):
        return cls(start_timestamp=[],
                   end_timestamp=[],
                   train_loss=[],
                   val_loss=[],
                   metrics={name: [] for metric in metrics for name in metric.names})

    @classmethod
    def load(cls, path: PathLike):
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
            start_timestamp = [datetime.fromisoformat(ts) for ts in data['start_timestamp']]
            end_timestamp = [datetime.fromisoformat(ts) for ts in data['end_timestamp']]
            train_loss = data['train_loss']
            val_loss = data['val_loss']
            metrics = data['metrics']
        return cls(start_timestamp=start_timestamp,
                   end_timestamp=end_timestamp,
                   train_loss=train_loss,
                   val_loss=val_loss,
                   metrics=metrics)

    def save(self, path: PathLike):
        path = Path(path)
        assert path.suffix == '.json'
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        start_timestamp = [str(ts) for ts in self.start_timestamp]
        end_timestamp = [str(ts) for ts in self.end_timestamp]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'train_loss': self.train_loss,
                'val_loss': self.val_loss,
                'metrics': self.metrics
            }, f, ensure_ascii=False)

    def add(self, start_timestamp: datetime, end_timestamp: datetime, train_loss: float, val_loss: float,
            metrics: Dict[str, float]):
        assert metrics.keys() == self.metrics.keys()
        assert start_timestamp.tzinfo == timezone.utc
        assert end_timestamp.tzinfo == timezone.utc
        self.start_timestamp.append(start_timestamp)
        self.end_timestamp.append(end_timestamp)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        for key in metrics:
            self.metrics[key].append(metrics[key])

    def __len__(self) -> int:
        assert len(self.train_loss) == 0 or self.train_loss[0] is None
        return max(0, len(self.train_loss) - 1)

    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def epoch(self) -> int:
        return len(self)
