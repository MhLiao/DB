import cv2
import numpy as np

from concern.config import Configurable, State
import concern.webcv2 as webcv2
from .data_process import DataProcess


class _ResizeImage:
    '''
    Resize images.
    Inputs:
        image_size: two-tuple-like object (height, width).
        mode: the mode used to resize image. Valid options:
            "keep_size": keep the original size of image.
            "resize":    arbitrarily resize the image to image_size.
            "keep_ratio": resize to dest height
                while keepping the height/width ratio of the input.
            "pad": pad the image to image_size after applying
                "keep_ratio" resize.
    '''
    MODES = ['resize', 'keep_size', 'keep_ratio', 'pad']

    def __init__(self, image_size, mode):
        self.image_size = image_size
        assert mode in self.MODES
        self.mode = mode

    def resize_or_pad(self, image):
        if self.mode == 'keep_size':
            return image
        if self.mode == 'pad':
            return self.pad_iamge(image)

        assert self.mode in ['resize', 'keep_ratio']
        height, width = self.get_image_size(image)
        image = cv2.resize(image, (width, height))
        return image

    def get_image_size(self, image):
        height, width = self.image_size
        if self.mode == 'keep_ratio':
            width = max(width, int(
                height / image.shape[0] * image.shape[1] / 32 + 0.5) * 32)
        if self.mode == 'pad':
            width = min(width,
                        max(int(height / image.shape[0] * image.shape[1] / 32 + 0.5) * 32, 32))
        return height, width

    def pad_iamge(self, image):
        canvas = np.zeros((*self.image_size, 3), np.float32)
        height, width = self.get_image_size(image)
        image = cv2.resize(image, (width, height))
        canvas[:, :width, :] = image
        return canvas


class ResizeImage(_ResizeImage, DataProcess):
    mode = State(default='keep_ratio')
    image_size = State(default=[1152, 2048])  # height, width
    key = State(default='image')

    def __init__(self, cmd={}, mode=None, **kwargs):
        self.load_all(**kwargs)
        if mode is not None:
            self.mode = mode
        if 'resize_mode' in cmd:
            self.mode = cmd['resize_mode']
        assert self.mode in self.MODES

    def process(self, data):
        data[self.key] = self.resize_or_pad(data[self.key])
        return data


class ResizeData(_ResizeImage, DataProcess):
    key = State(default='image')
    box_key = State(default='polygons')
    image_size = State(default=[64, 256])  # height, width

    def __init__(self, cmd={}, mode=None, key=None, box_key=None, **kwargs):
        self.load_all(**kwargs)
        if mode is not None:
            self.mode = mode
        if key is not None:
            self.key = key
        if box_key is not None:
            self.box_key = box_key
        if 'resize_mode' in cmd:
            self.mode = cmd['resize_mode']
        assert self.mode in self.MODES

    def process(self, data):
        height, width = data['image'].shape[:2]
        new_height, new_width = self.get_image_size(data['image'])
        data[self.key] = self.resize_or_pad(data[self.key])

        charboxes = data[self.box_key]
        data[self.box_key] = charboxes.copy()
        data[self.box_key][:, :, 0] = data[self.box_key][:, :, 0] * \
            new_width / width
        data[self.box_key][:, :, 1] = data[self.box_key][:, :, 1] * \
            new_height / height
        return data
