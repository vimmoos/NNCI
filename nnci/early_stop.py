from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    metric: callable
    patience: int = 4
    inverted: bool = False
    cnt: int = field(init=False, default_factory=lambda: 0)
    min_val: float = field(init=False, default_factory=lambda: float("inf"))

    @property
    def removable(self):
        return -self.cnt if self.cnt >= 1 else None

    def __call__(self, pred, target):
        metric = self.metric(pred, target) * (-1 if self.inverted else 1)
        if metric < self.min_val:
            self.min_val = metric
            self.cnt = 0
            return False, metric

        if metric >= self.min_val:
            self.cnt += 1

        return self.cnt >= self.patience, metric
