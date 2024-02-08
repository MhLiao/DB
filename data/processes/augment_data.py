import imgaug
import numpy as np

from concern.config import State
from .data_process import DataProcess
from data.augmenter import AugmenterBuilder
import cv2
import math
import albumentations as A
import shapely

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
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly

class AugmentAlbumentation(DataProcess):
    def __init__(self, **kwargs):
        self.only_resize = kwargs.get('only_resize')

        self.augfn = A.Compose([
            A.VerticalFlip(p=0.5 ),
            A.HorizontalFlip(p=0.5 ),
            # A.Transpose(p=0.5 ),
            # A.RandomRotate90(p=0.5 ),
            A.ShiftScaleRotate(shift_limit=0.1,
                            scale_limit=0.0,
                            value = (0,0,0),
                            rotate_limit=15,
                            p=0.5,
                            border_mode = cv2.BORDER_CONSTANT),
            A.PadIfNeeded(
                min_height=640,
                min_width=640,
                value = (0,0,0),
                border_mode=cv2.BORDER_CONSTANT),
            A.RandomCrop(height=640, width=640, p=1.0),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
            ], p=0.7),
            #A.OneOf([
            #    A.OpticalDistortion(distort_limit=1.0),
            #    A.GridDistortion(num_steps=5, distort_limit=1.),
            #    A.Affine(shear = 3, p=1.),
            #    A.ElasticTransform(alpha=2),
            #], p=0.7),
            # A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def process(self, data):
        image = data['image'].astype(np.uint8)
        # Convert polygon to keypoints. Ref: https://github.com/albumentations-team/albumentations/issues/750
        keypoints = np.array([line['poly'] for line in data['lines']]) # shape of [n_poly, poly_points, 2]
        n_points_list = [len(it) for it in keypoints] # remember poly_points's length
        keypoints = keypoints.reshape(-1, 2).tolist() # flatten poly to 2d array shape of [n_poly * poly_points, 2]
        keypoints = [val + [i] for i, val in enumerate(keypoints)] # remember the order of points

        
        # resize image base on image size and polygon size
        bboxes = [line['poly'] for line in data['lines'] if len(line['poly']) == 4]
        if len(bboxes) > 0:
            bboxes_height = [min(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])) for box in bboxes]
            lower_bound = max(0.5, 10 / min(bboxes_height)) # text height must be over than 20 pixel
            upper_bound = min(5, 200 / max(bboxes_height)) # text height must be less than

            resize_func = A.Compose([A.RandomScale(p=0.9, scale_limit=(lower_bound - 1, upper_bound - 1))], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
            
            # execute resizing
            transformed = resize_func(image=image, keypoints=keypoints)
            image, keypoints = transformed['image'], transformed['keypoints']
        # execute other aug func
        transformed = self.augfn(image=image, keypoints=keypoints)


        image = transformed['image'].astype(np.float32)
        keypoints = [i[:-1] for i in sorted(transformed['keypoints'], key=lambda x: x[2])] # remove the order after transformed
        # keypoints = [[max(min(x[0], image.shape[1]-1), 0), max(min(x[1], image.shape[0]-1), 0)] for x in keypoints_trans] # clip the point's coordinate being outside of img
        
        data['image'] = image
        line_polys = []
        point_i = 0
        for poly_i, n_points in enumerate(n_points_list):
            poly = keypoints[point_i : point_i + n_points]
            shapely_polygon = shapely.geometry.Polygon(poly)
            shapely_image = shapely.geometry.box(0, 0, image.shape[1], image.shape[0])
            intersect = shapely_image.intersection(shapely_polygon)
            if intersect.area / shapely_polygon.area > 0.1:
                poly = tuple(intersect.exterior.coords)
                line_polys.append({
                    'points': poly,
                    'ignore': data['lines'][poly_i]['text'] == '###',
                    'text': data['lines'][poly_i]['text'],
                })
            point_i += n_points
        data['polys'] = line_polys
        
        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=data['image'].shape[:2])

        if not self.only_resize:
            data['is_training'] = True 
        else:
            data['is_training'] = False

        return data
