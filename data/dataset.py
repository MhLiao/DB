from torch.utils.data import Dataset as TorchDataset

from concern.config import Configurable, State


class SliceDataset(TorchDataset, Configurable):
    dataset = State()
    start = State()
    end = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        if self.start is None:
            self.start = 0
        if self.end is None:
            self.end = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

    def __len__(self):
        return self.end - self.start
