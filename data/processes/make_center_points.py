import numpy as np

from concern.config import State
from .data_process import DataProcess


class MakeCenterPoints(DataProcess):
    box_key = State(default='charboxes')
    size = State(default=32)

    def process(self, data):
        shape = data['image'].shape[:2]
        points = np.zeros((self.size, 2), dtype=np.float32)
        boxes = np.array(data[self.box_key])[:self.size]

        size = boxes.shape[0]
        points[:size] = boxes.mean(axis=1)
        data['points'] = (points / shape[::-1]).astype(np.float32)
        return data
