import torch
import numpy as np


class Quad:
    def __init__(self, points, format='NP2'):
        self._rect = None
        self.tensorized = False
        self._points = None
        self.set_points(points, format)

    @property
    def points(self):
        return self._points

    def set_points(self, new_points, format='NP2'):
        order = (format.index('N'), format.index('P'), format.index('2'))

        if isinstance(new_points, torch.Tensor):
            self._points = new_points.permute(*order)
            self.tensorized = True
        else:
            points = np.array(new_points, dtype=np.float32)
            self._points = points.transpose(*order)

            if self.tensorized:
                self.tensorized = False
                self.tensor

    @points.setter
    def points(self, new_points):
        self.set_points(new_points)

    @property
    def tensor(self):
        if not self.tensorized:
            self._points = torch.from_numpy(self._points)
        return self._points

    def to(self, device):
        self._points.to(device)
        return self._points

    def __iter__(self):
        for i in range(self._points.shape[0]):
            if self.tensorized:
                yield self.tensor[i]
            else:
                yield self.points[i]


    def rect(self):
        if self._rect is None:
            self._rect = self.rectify()
        return self._rect

    def __getitem__(self, *args, **kwargs):
        return self._points.__getitem__(*args, **kwargs)

    def numpy(self):
        if not self.tensorized:
            return self._points
        return self._points.cpu().data.numpy()

    def rectify(self):
        if self.tensorized:
            return self.rectify_tensor()

        xmin = self._points[:, :, 0].min(axis=1)
        ymin = self._points[:, :, 1].min(axis=1)
        xmax = self._points[:, :, 0].max(axis=1)
        ymax = self._points[:, :, 1].max(axis=1)
        return np.stack([xmin, ymin, xmax, ymax], axis=1)

    def rectify_tensor(self):
        xmin, _ = self.tensor[:, :, 0].min(dim=1, keepdim=True)
        ymin, _ = self.tensor[:, :, 1].min(dim=1, keepdim=True)
        xmax, _ = self.tensor[:, :, 0].max(dim=1, keepdim=True)
        ymax, _ = self.tensor[:, :, 1].max(dim=1, keepdim=True)
        return torch.cat([xmin, ymin, xmax, ymax], dim=1)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self._points.__getattribute__(name)
