from bisect import bisect_right
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler

from concern.config import Configurable, State
from concern.signal_monitor import SignalMonitor


class ConstantLearningRate(Configurable):
    lr = State(default=0.0001)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_learning_rate(self, epoch, step):
        return self.lr


class FileMonitorLearningRate(Configurable):
    file_path = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        self.monitor = SignalMonitor(self.file_path)

    def get_learning_rate(self, epoch, step):
        signal = self.monitor.get_signal()
        if signal is not None:
            return float(signal)
        return None


class PriorityLearningRate(Configurable):
    learning_rates = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_learning_rate(self, epoch, step):
        for learning_rate in self.learning_rates:
            lr = learning_rate.get_learning_rate(epoch, step)
            if lr is not None:
                return lr
        return None


class MultiStepLR(Configurable):
    lr = State()
    milestones = State(default=[])  # milestones must be sorted
    gamma = State(default=0.1)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.lr = cmd.get('lr', self.lr)

    def get_learning_rate(self, epoch, step):
        return self.lr * self.gamma ** bisect_right(self.milestones, epoch)


class WarmupLR(Configurable):
    steps = State(default=4000)
    warmup_lr = State(default=1e-5)
    origin_lr = State()

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)

    def get_learning_rate(self, epoch, step):
        if epoch == 0 and step < self.steps:
            return self.warmup_lr
        return self.origin_lr.get_learning_rate(epoch, step)


class PiecewiseConstantLearningRate(Configurable):
    boundaries = State(default=[10000, 20000])
    values = State(default=[0.001, 0.0001, 0.00001])

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_learning_rate(self, epoch, step):
        for boundary, value in zip(self.boundaries, self.values[:-1]):
            if step < boundary:
                return value
        return self.values[-1]


class DecayLearningRate(Configurable):
    lr = State(default=0.007)
    epochs = State(default=1200)
    factor = State(default=0.9)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_learning_rate(self, epoch, step=None):
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr


class BuitlinLearningRate(Configurable):
    lr = State(default=0.001)
    klass = State(default='StepLR')
    args = State(default=[])
    kwargs = State(default={})

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.lr = cmd.get('lr', None) or self.lr
        self.scheduler = None

    def prepare(self, optimizer):
        self.scheduler = getattr(lr_scheduler, self.klass)(
            optimizer, *self.args, **self.kwargs)

    def get_learning_rate(self, epoch, step=None):
        if self.scheduler is None:
            raise 'learning rate not ready(prepared with optimizer) '
        self.scheduler.last_epoch = epoch
        # return value of gt_lr is a list,
        # where each element is the corresponding learning rate for a
        # paramater group.
        return self.scheduler.get_lr()[0]
