import warnings
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from concern.config import State
from .data_process import DataProcess


class MakeCenterDistanceMap(DataProcess):
    r'''
    Making the border map from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    expansion_ratio = State(default=0.1)

    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
        warnings.simplefilter("ignore")

    def process(self, data, *args, **kwargs):
        r'''
        required keys:
            image.
            lines: Instace of `TextLines`, which is defined in data/text_lines.py
        adding keys:
            distance_map
        '''
        image = data['image']
        lines = data['lines']

        h, w = image.shape[:2]
        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        for _, quad in lines:
            padded = self.expand_quad(quad)
            center_x = padded[:, 0].mean()
            center_y = padded[:, 1].mean()
            index_x, index_y = np.meshgrid(np.arange(w), np.arange(h))
            self.render_distance_map(canvas, center_x, center_y, index_x, index_y)
            self.render_constant(mask, quad, 1)

        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        return data

    def expand_quad(self, polygon):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.expansion_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        return padded_polygon
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)




    def distance(self, xs, ys, point):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2


