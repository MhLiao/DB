import numpy as np
import scipy.ndimage.filters as fi

from concern.config import State

from .data_process import DataProcess


class MakeCenterMap(DataProcess):
    max_size = State(default=32)
    shape = State(default=(64, 256))
    sigma_ratio = State(default=16)
    function_name = State(default='sample_gaussian')
    points_key = 'points'
    correlation = 0  # The formulation of guassian is simplified when correlation is 0

    def process(self, data):
        assert self.points_key in data, '%s in data is required' % self.points_key
        points = data['points'] * self.shape[::-1]  # N, 2
        assert points.shape[0] >= self.max_size
        func = getattr(self, self.function_name)
        data['charmaps'] = func(points, *self.shape)
        return data

    def gaussian(self, points, height, width):
        index_x, index_y = np.meshgrid(np.linspace(0, width, width),
                                       np.linspace(0, height, height))
        index_x = np.repeat(index_x[np.newaxis], points.shape[0], axis=0)
        index_y = np.repeat(index_y[np.newaxis], points.shape[0], axis=0)
        mu_x = points[:, 0][:, np.newaxis, np.newaxis]
        mu_y = points[:, 1][:, np.newaxis, np.newaxis]
        mask_is_zero = ((mu_x == 0) + (mu_y == 0)) == 0
        result = np.reciprocal(2 * np.pi * width / self.sigma_ratio * height / self.sigma_ratio)\
            * np.exp(- 0.5 * (np.square((index_x - mu_x) / width * self.sigma_ratio) +
                              np.square((index_y - mu_y) / height * self.sigma_ratio)))

        result = result / \
            np.maximum(result.max(axis=1, keepdims=True).max(
                axis=2, keepdims=True), np.finfo(np.float32).eps)
        result = result * mask_is_zero
        return result.astype(np.float32)

    def sample_gaussian(self, points, height, width):
        points = (points + 0.5).astype(np.int32)
        canvas = np.zeros((self.max_size, height, width), dtype=np.float32)
        for index in range(canvas.shape[0]):
            point = points[index]
            canvas[index, point[1], point[0]] = 1.
            if point.sum() > 0:
                fi.gaussian_filter(canvas[index], (height // self.sigma_ratio,
                                                   width // self.sigma_ratio),
                                   output=canvas[index], mode='mirror')
                canvas[index] = canvas[index] / canvas[index].max()
                x_range = min(point[0], width - point[0])
                canvas[index, :, :point[0] - x_range] = 0
                canvas[index, :, point[0] + x_range:] = 0
                y_range = min(point[1], width - point[1])
                canvas[index, :point[1] - y_range, :] = 0
                canvas[index, point[1] + y_range:, :] = 0
        return canvas
