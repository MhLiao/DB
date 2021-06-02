import imgaug
import numpy as np

from concern.config import State
from .data_process import DataProcess
from data.augmenter import AugmenterBuilder
import cv2
import math


class AugmentData(DataProcess):
    augmenter_args = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.keep_ratio = kwargs.get('keep_ratio')
        self.only_resize = kwargs.get('only_resize')
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True 
        else:
            data['is_training'] = False 
        return data


class AugmentDetectionData(AugmentData):
    
    def may_augment_annotation(self, aug: imgaug.augmenters.Augmenter, data, shape):
        if aug is None:
            return data
        
        line_polys = []
        keypoints = []
        texts = []
        for line in data['lines']:
            texts.append(line['text'])
            for p in line['poly']:
                keypoints.append(imgaug.Keypoint(p[0], p[1]))
        
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints=keypoints, shape=shape)])[0].keypoints
        new_polys = np.array([[p.x, p.y] for p in keypoints]).reshape((-1, 4, 2))
    
        for i in range(len(texts)):
            poly = new_polys[i]
            line_polys.append({
                'points': poly,
                'ignore': texts[i] == '###',
                'text': texts[i]
            })
        
        data['polys'] = line_polys

