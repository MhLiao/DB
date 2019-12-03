import pickle

import cv2
import skimage
import numpy as np
from shapely.geometry import Polygon

from concern.config import Configurable, State


def binary_search_smallest_width(poly):
    if len(poly) < 3:
        return 0
    poly = Polygon(poly)
    low = 0
    high = 65536
    while high - low > 0.1:
        mid = (high + low) / 2
        mid_poly = poly.buffer(-mid)
        if mid_poly.geom_type == 'Polygon' and mid_poly.area > 0.1:
            low = mid
        else:
            high = mid
    height = (low + high) / 2
    if height < 0.1:
        return 0
    else:
        return height


def project_point_to_line(x, u, v, axis=0):
    n = v - u
    n = n / (np.linalg.norm(n, axis=axis, keepdims=True) + np.finfo(np.float32).eps)
    p = u + n * np.sum((x - u) * n, axis=axis, keepdims=True)
    return p


def project_point_to_segment(x, u, v, axis=0):
    p = project_point_to_line(x, u, v, axis=axis)
    outer = np.greater_equal(np.sum((u - p) * (v - p), axis=axis, keepdims=True), 0)
    near_u = np.less_equal(
        np.linalg.norm(u - p, axis=axis, keepdims=True),
        np.linalg.norm(v - p, axis=axis, keepdims=True)
    )
    o = np.where(outer, np.where(near_u, u, v), p)
    return o


class MakeSimpleDetectionData(Configurable):
    center_shrink = State(default=0.5)
    background_weight = State(default=3.0)

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def get_mask(self, w, h, polys, ignores):
        mask = np.ones((h, w), np.float32)

        for poly, ignore in zip(polys, ignores):
            if ignore:
                cv2.fillPoly(mask, np.array([poly], np.int32), 0.0)

        return mask

    def get_line_height(self, poly):
        return binary_search_smallest_width(poly)

    def get_regions_coords(self, w, h, polys, heights, shrink):
        label_map = np.zeros((h, w), np.int32)
        for line_id, (poly, height) in enumerate(zip(polys, heights)):
            if height > 0:
                shrinked_poly = Polygon(poly).buffer(-height * shrink)
                if shrinked_poly.geom_type == 'Polygon' and not shrinked_poly.is_empty:
                    shrinked_poly = np.array(list(shrinked_poly.exterior.coords), np.int32)
                    cv2.fillPoly(label_map, shrinked_poly[np.newaxis], line_id + 1)

        regions = skimage.measure.regionprops(label_map)
        regions_coords = [
            region.coords[:, ::-1] for region in regions
        ] + [
            np.zeros((0, 2), 'int32')
        ] * (len(polys) - len(regions))

        return regions_coords

    def get_coords_poly_projection(self, coords, poly):
        start_points = np.array(poly)
        end_points = np.concatenate([poly[1:], poly[:1]], axis=0)
        region_points = coords

        projected_points = project_point_to_segment(
            region_points[:, np.newaxis],
            start_points[np.newaxis],
            end_points[np.newaxis],
            axis=2,
        )
        projection_distances = np.linalg.norm(region_points[:, np.newaxis] - projected_points, axis=2)
        best_projected_points = projected_points[np.arange(len(region_points)), np.argmin(projection_distances, axis=1)]
        return best_projected_points

    def get_coords_poly_distance(self, coords, poly):
        projection = self.get_coords_poly_projection(coords, poly)
        return np.linalg.norm(projection - coords, axis=1)

    def get_normalized_weight(self, heatmap, mask):
        pos = np.greater_equal(heatmap, 0.5)
        neg = 1 - pos
        pos = np.logical_and(pos, mask)
        neg = np.logical_and(neg, mask)
        npos = pos.sum()
        nneg = neg.sum()
        smooth = (npos + nneg + 1) * 0.05
        wpos = (nneg + smooth) / (npos + smooth)
        weight = np.zeros_like(heatmap)
        weight[neg] = self.background_weight
        weight[pos] = wpos
        return weight

    def draw_maps(self, w, h, polys, ignores):
        raise NotImplementedError()

    def __call__(self, data, *args, **kwargs):
        image, label, meta = data
        lines = label['polys']

        h, w = image.shape[:2]

        polys = []
        ignores = []
        for line in lines:
            if len(line['points']) >= 4:
                polys.append(line['points'])
                ignores.append(line['ignore'])

        maps = self.draw_maps(w, h, polys, ignores)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        label = maps
        return image, label, pickle.dumps(meta)


class MakeSimpleSegData(MakeSimpleDetectionData):
    def draw_maps(self, w, h, polys, ignores):
        heatmap = np.zeros((h, w), np.float32)

        heights = [self.get_line_height(poly) for poly in polys]
        regions_center_coords = self.get_regions_coords(w, h, polys, heights, self.center_shrink)
        train_mask = self.get_mask(w, h, polys, ignores)
        for region_center_coords in regions_center_coords:
            x, y = region_center_coords[:, 0], region_center_coords[:, 1]
            heatmap[y, x] = 1.0
        heatmap_weight = self.get_normalized_weight(heatmap, train_mask)

        return {
            'heatmap': heatmap[np.newaxis],
            'heatmap_weight': heatmap_weight[np.newaxis],
        }


class MakeSimpleEASTData(MakeSimpleDetectionData):
    def draw_maps(self, w, h, polys, ignores):
        heatmap = np.zeros((h, w), np.float32)
        densebox = np.zeros((8, h, w), np.float32)
        densebox_weight = np.zeros((h, w), np.float32)

        heights = [self.get_line_height(poly) for poly in polys]
        regions_center_coords = self.get_regions_coords(w, h, polys, heights, self.center_shrink)
        train_mask = self.get_mask(w, h, polys, ignores)
        for poly, region_center_coords in zip(polys, regions_center_coords):
            x, y = region_center_coords[:, 0], region_center_coords[:, 1]
            heatmap[y, x] = 1.0
            densebox_weight[y, x] = 1.0

            for i in range(0, 4):
                densebox[i * 2, y, x] = float(poly[i][0]) - x
                densebox[i * 2 + 1, y, x] = float(poly[i][1]) - y

        heatmap_weight = self.get_normalized_weight(heatmap, train_mask)
        densebox_weight = densebox_weight * train_mask

        return {
            'heatmap': heatmap[np.newaxis],
            'heatmap_weight': heatmap_weight[np.newaxis],
            'densebox': densebox,
            'densebox_weight': densebox_weight[np.newaxis],
        }


class MakeSimpleTextsnakeData(MakeSimpleDetectionData):
    def draw_maps(self, w, h, polys, ignores):
        heatmap = np.zeros((h, w), np.float32)
        radius = np.zeros((h, w), np.float32)
        radius_weight = np.zeros((h, w), np.float32)

        heights = [self.get_line_height(poly) for poly in polys]
        regions_center_coords = self.get_regions_coords(w, h, polys, heights, self.center_shrink)
        train_mask = self.get_mask(w, h, polys, ignores)
        for poly, region_center_coords in zip(polys, regions_center_coords):
            x, y = region_center_coords[:, 0], region_center_coords[:, 1]
            heatmap[y, x] = 1.0

            distance = self.get_coords_poly_distance(region_center_coords, poly)
            radius[y, x] = distance
            radius_weight[y, x] = 1.0

        heatmap_weight = self.get_normalized_weight(heatmap, train_mask)
        radius_weight = radius_weight * train_mask

        return {
            'heatmap': heatmap[np.newaxis],
            'heatmap_weight': heatmap_weight[np.newaxis],
            'radius': radius[np.newaxis],
            'radius_weight': radius_weight[np.newaxis],
        }


class MakeSimpleMSRData(MakeSimpleDetectionData):
    def draw_maps(self, w, h, polys, ignores):
        heatmap = np.zeros((h, w), np.float32)
        offset = np.zeros((2, h, w), np.float32)
        offset_weight = np.zeros((h, w), np.float32)

        heights = [self.get_line_height(poly) for poly in polys]
        regions_center_coords = self.get_regions_coords(w, h, polys, heights, self.center_shrink)
        train_mask = self.get_mask(w, h, polys, ignores)
        for poly, region_center_coords in zip(polys, regions_center_coords):
            x, y = region_center_coords[:, 0], region_center_coords[:, 1]
            heatmap[y, x] = 1.0

            projection_points = self.get_coords_poly_projection(region_center_coords, poly)
            offset[0, y, x] = projection_points[:, 0] - x
            offset[1, y, x] = projection_points[:, 1] - y
            offset_weight[y, x] = 1.0

        heatmap_weight = self.get_normalized_weight(heatmap, train_mask)
        offset_weight = offset_weight * train_mask

        return {
            'heatmap': heatmap[np.newaxis],
            'heatmap_weight': heatmap_weight[np.newaxis],
            'offset': offset,
            'offset_weight': offset_weight[np.newaxis],
        }


class SimpleDetectionCropper(Configurable):
    patch_cropper = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    def crop(self, batch, output):
        img, label, meta = batch

        images_polys = []
        images_patches = []
        for polys, image_meta in zip(output['polygons_pred'], meta):
            image_meta = pickle.loads(image_meta)

            images_polys.append(polys)
            images_patches.append([self.patch_cropper.crop(image_meta['image'], p) for p in polys])

        return images_polys, images_patches
