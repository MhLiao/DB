import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

from concern.config import State
from .data_process import DataProcess


class MakeSegDetectionData(DataProcess):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=8)
    shrink_ratio = State(default=0.4)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']
        lines = data['lines']

        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            # height = min(np.linalg.norm(polygon[0] - polygon[3]),
            #              np.linalg.norm(polygon[1] - polygon[2]))
            # width = min(np.linalg.norm(polygon[0] - polygon[1]),
            #             np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                # polygon_shape, polygon = self.process_with_convexHull(image, polygon)
                # polygons[i] = polygon
                polygon_shape = Polygon(polygon)
                # if lines[i]["text"] != "1":
                if True:
                    distance = polygon_shape.area * \
                        (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                else:
                    distance = polygon_shape.area * \
                        (1 - np.power(0.7, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)

        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, filename=filename)
        return data

    def process_with_convexHull(self, image, polygon):
        try:
            polygon = polygon.astype(np.int32)
            x_min, y_min, x_max, y_max = min(polygon[:,0]), min(polygon[:, 1]), max(polygon[:, 0]), max(polygon[:, 1])
            piece = image[y_min:y_max, x_min:x_max, :].astype(np.uint8)
            _, binary = cv2.threshold(cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # image = cv2.polylines(image, [polygon], True, (0, 255, 0), thickness=2)
            hull = cv2.convexHull(np.vstack(contours))
            # cv2.drawContours(piece, [hull], 0, (0, 255, 0))
            # cv2.imwrite("hoge.png", piece)
            # exit()
            hull[:, :, 0] += x_min
            hull[:, :, 1] += y_min
            polygon_shape = Polygon(hull.reshape(-1, 2))
            return polygon_shape, hull.reshape(-1, 2)
        except:
            return Polygon(polygon), polygon

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

