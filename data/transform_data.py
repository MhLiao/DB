import numpy as np
import torch

from concern.config import Configurable


class TransformData(Configurable):
    '''
    this transformation is inplcae, which means that the input
        will be modified.
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def __call__(self, data_dict, *args, **kwargs):
        image = data_dict['image'].transpose(2, 0, 1)
        image = image / 255.0
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        data_dict['image'] = image
        return data_dict
