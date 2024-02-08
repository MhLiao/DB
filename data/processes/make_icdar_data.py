from collections import OrderedDict

import torch
import numpy as np

from concern.config import Configurable, State
from .data_process import DataProcess
import cv2


class MakeICDARData(DataProcess):
    shrink_ratio = State(default=0.4)

    def __init__(self, debug=False, cmd={}, **kwargs):
        self.load_all(**kwargs)

        self.debug = debug
        if 'debug' in cmd:
            self.debug = cmd['debug']

    def process(self, data):
        polygons = []
        ignore_tags = []
        annotations = data['polys']
        for annotation in annotations:
            polygons.append(np.array(annotation['points']))
            # polygons.append(annotation['points'])
            ignore_tags.append(annotation['ignore'])
        ignore_tags = np.array(ignore_tags, dtype=np.uint8)
        filename = data.get('filename', data['data_id'])
        if self.debug:
            self.draw_polygons(data['image'], polygons, ignore_tags)
        shape = np.array(data['shape'])
        return OrderedDict(image=data['image'],
                           polygons=polygons,
                           ignore_tags=ignore_tags,
                           shape=shape,
                           filename=filename,
                           is_training=data['is_training'],
                           lines=data['lines'])

    def draw_polygons(self, image, polygons, ignore_tags):
        for i in range(len(polygons)):
            polygon = polygons[i].reshape(-1, 2).astype(np.int32)
            ignore = ignore_tags[i]
            if ignore:
                color = (255, 0, 0)  # depict ignorable polygons in blue
            else:
                color = (0, 0, 255)  # depict polygons in red

            cv2.polylines(image, [polygon], True, color, 1)
    polylines = staticmethod(draw_polygons)


class ICDARCollectFN(Configurable):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = OrderedDict()
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                data_dict[k].append(v)
        data_dict['image'] = torch.stack(data_dict['image'], 0)
        return data_dict

