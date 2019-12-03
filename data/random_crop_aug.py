import random
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

from concern.config import Configurable, State


def regular_resize(image, boxes, tags, crop_size):
    h, w, c = image.shape
    if h < w:
        scale_ratio = crop_size * 1.0 / w
        new_h = int(round(crop_size * h * 1.0 / w))
        if new_h > crop_size:
            new_h = crop_size
        image = cv2.resize(image, (crop_size, new_h))
        new_img = np.zeros((crop_size, crop_size, 3))
        new_img[:new_h, :, :] = image
        boxes *= scale_ratio
    else:
        scale_ratio = crop_size * 1.0 / h
        new_w = int(round(crop_size * w * 1.0 / h))
        if new_w > crop_size:
            new_w = crop_size
        image = cv2.resize(image, (new_w, crop_size))
        new_img = np.zeros((crop_size, crop_size, 3))
        new_img[:, :new_w, :] = image
        boxes *= scale_ratio
    return new_img, boxes, tags


def random_crop(image, boxes, tags, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio):
    h, w, c = image.shape
    selected_boxes = []
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy)
        ymax = np.max(yy)
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < min_crop_side_ratio*w or ymax - ymin < min_crop_side_ratio*h:
            # area too small
            continue
        if boxes.shape[0] != 0:
            box_axis_in_area = (boxes[:, :, 0] >= xmin) & (boxes[:, :, 0] <= xmax) \
                & (boxes[:, :, 1] >= ymin) & (boxes[:, :, 1] <= ymax)
            selected_boxes = np.where(np.sum(box_axis_in_area, axis=1) == 4)[0]
            if len(selected_boxes) > 0:
                if (tags[selected_boxes] == False).astype(np.float).sum() > 0:
                    break
        else:
            selected_boxes = []
            break
    if i == max_tries - 1:
        return regular_resize(image, boxes, tags, crop_size)

    image = image[ymin:ymax+1, xmin:xmax+1, :]
    boxes = boxes[selected_boxes]
    tags = tags[selected_boxes]
    boxes[:, :, 0] -= xmin
    boxes[:, :, 1] -= ymin
    return regular_resize(image, boxes, tags, crop_size)


def regular_crop(image, boxes, tags, crop_size, max_tries, w_array, h_array, w_axis, h_axis, min_crop_side_ratio):
    h, w, c = image.shape
    mask_w = np.arange(w - crop_size)
    mask_h = np.arange(h - crop_size)
    keep_w = np.where(np.logical_and(
        w_array[mask_w] == 0, w_array[mask_w + crop_size - 1] == 0))[0]
    keep_h = np.where(np.logical_and(
        h_array[mask_h] == 0, h_array[mask_h + crop_size - 1] == 0))[0]

    if keep_w.size > 0 and keep_h.size > 0:
        for i in range(max_tries):
            xmin = np.random.choice(keep_w, size=1)[0]
            xmax = xmin + crop_size
            ymin = np.random.choice(keep_h, size=1)[0]
            ymax = ymin + crop_size
            if boxes.shape[0] != 0:
                box_axis_in_area = (boxes[:, :, 0] >= xmin) & (boxes[:, :, 0] <= xmax) \
                    & (boxes[:, :, 1] >= ymin) & (boxes[:, :, 1] <= ymax)
                selected_boxes = np.where(
                    np.sum(box_axis_in_area, axis=1) == 4)[0]
                if len(selected_boxes) > 0:
                    if (tags[selected_boxes] == False).astype(np.float).sum() > 0:
                        break
            else:
                selected_boxes = []
                break
        if i == max_tries-1:
            return random_crop(image, boxes, tags, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio)
        image = image[ymin:ymax, xmin:xmax, :]
        boxes = boxes[selected_boxes]
        tags = tags[selected_boxes]
        boxes[:, :, 0] -= xmin
        boxes[:, :, 1] -= ymin
        return image, boxes, tags
    else:
        return random_crop(image, boxes, tags, crop_size, max_tries, w_axis, h_axis, min_crop_side_ratio)


class RandomCrop(object):
    def __init__(self, crop_size=640, max_tries=50, min_crop_side_ratio=0.1):
        self.crop_size = crop_size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, image, boxes, tags):
        h, w, _ = image.shape
        h_array = np.zeros((h), dtype=np.int32)
        w_array = np.zeros((w), dtype=np.int32)

        for box in boxes:
            box = np.round(box, decimals=0).astype(np.int32)
            minx = np.min(box[:, 0])
            maxx = np.max(box[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(box[:, 1])
            maxy = np.max(box[:, 1])
            h_array[miny:maxy] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            # resize image
            return regular_resize(image, boxes, tags, self.crop_size)

        if h <= self.crop_size + 1 or w <= self.crop_size + 1:
            return random_crop(image, boxes, tags, self.crop_size, self.max_tries, w_axis, h_axis, self.min_crop_side_ratio)
        else:
            return regular_crop(image, boxes, tags, self.crop_size, self.max_tries, w_array, h_array, w_axis, h_axis, self.min_crop_side_ratio)


class RandomCropAug(Configurable):
    size = State(default=640)

    def __init__(self, size=640, *args, **kwargs):
        self.size = size or self.size
        self.augment = RandomCrop(size)

    def __call__(self, data):
        '''
        This augmenter is supposed to following the process of `MakeICDARData`,
        in which labels are mapped to this specific format:
            (image, polygons: (n, 4, 2), tags: [Boolean], ...)
        '''
        image, boxes, ignore_tags = data[:3]
        image, boxes, ignore_tags = self.augment(image, boxes, ignore_tags)
        return (image, boxes, ignore_tags, *data[3:])
