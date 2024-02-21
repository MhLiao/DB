import argparse
import os
import torch
from torch import nn
import cv2
import numpy as np


import math
from glob import glob
from backbones.resnet import deformable_resnet18
from decoders.seg_detector_asf import SegSpatialScaleDetector
from decoders.seg_detector_loss import SegDetectorLossBuilder
from tqdm import tqdm
from scipy.signal import find_peaks
import itertools
from shapely.geometry import Polygon
import pyclipper

from time import time

RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
DEBUG=True
# number of chunks
DEF_chunk_nums = 10

def main():
    parser = argparse.ArgumentParser(description='Applying DBNet for multiline')
    parser.add_argument('--xml', type=str, default='experiments/dummy_resnet18.yaml', help='xml path to load the model')
    parser.add_argument('--resume', type=str, default='./final', help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./results/', help='path to save results')
    parser.add_argument('--image_short_side', type=int, default=0, help='set this arg as the portion of 32')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--pad', type=int, default=0,
                        help='add padding to the input image in all 4 directions: left, right, top, bottom')                   
    parser.add_argument('--cuda', action='store_true', default=False, help='using cuda if called, otherwise cpu')
    
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    

    if args['cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # load model
    model = SegDetectorModel(args=None, device=device)
    states = torch.load(
        args['resume'], map_location=device)

    model.load_state_dict(states, strict=False)
    model.eval()

    if os.path.isdir(args['image_path']):
        img_paths = [it for it in glob(f"{args['image_path']}/*.*") if "jpg" in it or "png" in it or "jpeg" in it]
    else:
        img_paths = [args['image_path']]
    os.makedirs(args['result_dir'], exist_ok=True)
    start = time()
    for path in tqdm(img_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if args['pad'] > 0:
            # get histogram
            hist = np.histogram(img, bins=2)

            hists = list(zip(hist[0], hist[1]))
            # sort by frequency
            hist_sort_by_frequency = sorted(hists, key=lambda hist: hist[0])
            # most pixel color as background color
            main_color = hist_sort_by_frequency[1][1]
            # second most pixel color as pen color
            pen_color = hist_sort_by_frequency[0][1]
            if main_color > pen_color:
                img = cv2.copyMakeBorder(img, args['pad'], args['pad'], args['pad'], args['pad'], cv2.BORDER_CONSTANT, None, (255, 255, 255))
            else:
                img = cv2.copyMakeBorder(img, args['pad'], args['pad'], args['pad'], args['pad'], cv2.BORDER_CONSTANT, None, (0, 0, 0))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = remove_background_imageset_by_opencv(gray)
        heat = CNN_forward(model, img.astype('float32'), args['image_short_side'])
        all_boxes, original_boxes = split_line_with_contour(gray, heat, args['box_thresh'])
        for idx, (box, ori) in enumerate(zip(all_boxes, original_boxes)):
            cv2.polylines(img, [box], True, (0, 255, 0), 1)
            min_x = np.argmin(box[:, 0])
            cv2.putText(img, str(idx), box[min_x, :], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # if ori is not None:
            #     cv2.polylines(img, [ori], True, (0, 0, 255), 1)
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args['result_dir'], os.path.basename(path)), np.hstack([heat, img]))
    print(time() - start)

class BasicModel(nn.Module):
    def __init__(self, device):
        nn.Module.__init__(self)
        super(BasicModel, self).__init__()
        self.device = device
        self.backbone = deformable_resnet18(pretrained=True)
        self.decoder = SegSpatialScaleDetector(in_channels=[64, 128, 256, 512], k=50,
                                    adaptive=True, attention_type='scale_channel_spatial')

    def forward(self, batch, *args, **kwargs):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        return self.decoder(self.backbone(data), *args, **kwargs)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(device)
        # for loading models
        self.model = nn.DataParallel(self.model)
        # self.model = self.model.module.to(device)
        self.criterion = SegDetectorLossBuilder('L1BalanceCELoss').build()
        # self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        
        data = data.float()
        pred = self.model.module(data, training=training)
        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred

def adjust_corner(contour, img_shape, kernel_size):
    binary = cv2.drawContours(np.zeros(img_shape, np.uint8), [contour], 0, 255, cv2.FILLED)
    kernel1 = np.zeros((kernel_size, kernel_size), np.uint8)
    kernel1[-1, :] = 1
    kernel1[:, -1] = 1
    kernel1[0, :] = 1
    kernel1[:, 0] = 1
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    # hoge = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
    # hoge[:, :, 0] = binary
    binary = cv2.dilate(binary, kernel1, iterations=1)
    binary = cv2.erode(binary, kernel2, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # hoge[:, :, 2] = binary
    # cv2.imwrite("hage.png", hoge)
    if len(contours) == 1:
        return contours[0]
    else:
        return contour


def check_min_rect_quality(min_rect, contour, overlap_ratio=0.8):
    p1 = Polygon(min_rect)
    p2 = Polygon(contour)
    if p2.area / p1.area > overlap_ratio:
        return True
    return False


def split_line_with_contour(img, heat, thresh):
    if np.max(heat) > thresh:
        heat = cv2.equalizeHist((heat * 255).astype(np.uint8)).astype(np.float32) / 255
    binary = heat > thresh
    binary = (binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in contours if min(cv2.boundingRect(contour)[2:]) > 5] # filter out <5 pixel height contour
    if len(contours) == 0: 
        # return boxes as whole image
        return [np.array([
            [0, 0],
            [img.shape[1], 0],
            [img.shape[1], img.shape[0]],
            [0, img.shape[0]]
        ])], [None]
    # max_height = max([cv2.boundingRect(contour)[3] for contour in contours])
    # contours_map = [cv2.drawContours(np.zeros(img.shape, np.uint8), [contour], 0, 255, cv2.FILLED) for contour in contours]

    # # determine how should we expand the contour to make them intersect
    # kernel = np.array([
    #     [1, 1, 1],
    #     [0, 1, 0],
    #     [1, 1, 1]
    # ], dtype=np.uint8) # dilate on vertical only
    # iterations_list = [9999] * len(contours)
    # print(len(contours))
    # #TODO: can optimize the speed here
    # for idx1, idx2 in itertools.combinations(range(len(contours)), 2):
    #     map1 = contours_map[idx1]
    #     map2 = contours_map[idx2]

    #     # check if 2 contours can overlap in x direction
    #     x1, y1, w1, h1 = cv2.boundingRect(contours[idx1])
    #     x2, y2, w2, h2 = cv2.boundingRect(contours[idx2])
    #     if min(x1 + w1, x2 + w2) - max(x1, x2) < -3:
    #         continue
    #     if min(y1 + h1, y2 + h2) - max(y1, y2) > -3 or min(y1 + h1, y2 + h2) - max(y1, y2) < - min(h1, h2):
    #         continue

    #     map12 = cv2.bitwise_or(map1, map2)
    #     for iteration in range(1, min(h1, h2, w1, w2)):
    #         map12 = cv2.dilate(map12, kernel, 1)
    #         contours_tmp, _ = cv2.findContours(map12, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours_tmp) < 2:
    #             iterations_list[idx1] = min(iterations_list[idx1], iteration - 1)
    #             iterations_list[idx2] = min(iterations_list[idx2], iteration - 1)
    #             break
    # # expand the contour right just before they intersect to each other
    # adjusted_contours = []
    # for contour, contour_map, iterations in zip(contours, contours_map, iterations_list):
    #     dilated = cv2.dilate(contour_map, np.ones((3,3)), iterations=iterations)
    #     contours_tmp, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours_tmp) < 1:
    #         assert "number of contour should be more than 1"
    #     adjusted_contours.append(contours_tmp[0])
    # contours = adjusted_contours
    # # hoge = np.dstack([img.copy()] * 3)
    # # cv2.drawContours(hoge, contours_tmp, -1, (0, 255, 0), 1)
    # # cv2.imwrite("hoge.png", hoge)
    # # exit()
    original_boxes = []
    all_boxes = []
    all_rects = []
    for contour in contours:
        # straight rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # piece = heat[y:y+h, x:x+w]
        if min(w, h) < 5:
            continue
        
        contour = adjust_corner(contour, img_shape=img.shape, kernel_size=h)

        # straight rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # piece = heat[y:y+h, x:x+w]
        if min(w, h) < 5:
            continue

        # skew rectangle
        bounding_box = cv2.minAreaRect(contour)
        # poly
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        min_rect = [points[index_1], points[index_2],
               points[index_3], points[index_4]]

        
        # approximated contour
        epsilon = 0.0002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = approx.reshape((-1, 2))
        if approx.shape[0] >= 4: # and not check_min_rect_quality(min_rect, approx):
            box = approx
            weight = 2.0
        else:
            box = min_rect
            weight = 2.0
        original_boxes.append(box)
        poly = Polygon(box)
        distance = poly.area * weight / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        box = np.array(offset.Execute(distance))
        # print(box)
        if len(box) > 1:
            original_boxes.pop()
            continue
        x, y, w, h = cv2.boundingRect(box)
        all_rects.append((x, y, w, h))
        all_boxes.append(box.reshape(-1, 2))
    idx = sorting_boxes(all_rects)
    all_boxes = list(map(lambda i: all_boxes[i], idx))
    # original_boxes = original_boxes[idx]
    return all_boxes, original_boxes



def sorting_boxes(boxes, overlap_thresh_y=0.5, overlap_thresh_x=0.5):
    boxes = np.array(boxes)
    idx_list = np.argsort(boxes[:, 1] + boxes[:, 3] / 2) # sort by y
    for i in range(len(idx_list) - 1):
        x1, y1, w1, h1 = boxes[idx_list[i]]
        for j in range(i + 1, len(idx_list)):
            x2, y2, w2, h2 = boxes[idx_list[j]]
            # check if these 2 boxes are overlap
            if (min(y1 + h1, y2 + h2) - max(y1, y2)) / min(h1, h2) > overlap_thresh_y and (min(x1 + w1, x2 + w2) - max(x1, x2)) / min(w1, w2) < overlap_thresh_x:
                if x1 > x2:
                    tmp = idx_list[i]
                    idx_list[i] = idx_list[j]
                    idx_list[j] = tmp
                    x1, y1, w1, h1 = x2, y2, w2, h2
            else:
                break
    return idx_list
            


def preprocess(img, image_short_side):
    '''
    return
        img as [C, H, W], float 32 from range 0~1
        original_shape before resize
    '''
    original_shape = img.shape[:2]
    img = resize_image(img, image_short_side)
    img -= RGB_MEAN
    img /= 255.
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, original_shape

def resize_image(img, image_short_side):
    height, width, _ = img.shape
    if image_short_side:
        if height < width:
            new_height = image_short_side
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = image_short_side
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
    else:
        new_width = (((width - 1) // 32) + 1) * 32
        new_height = (((height - 1) // 32) + 1) * 32
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def CNN_forward(model, img, image_short_side):
    torch_img, original_shape = preprocess(img, image_short_side)
    with torch.no_grad():
        # pred_backbone = model.backbone(torch_img)
        # pred = model.decoder(pred_backbone, training=True)
        pred = model.forward(torch_img, training=True)
    pred = pred['thresh_binary']
    pred = np.clip(pred[0, 0, ...].detach().cpu().numpy(), 0, 1)
    pred = cv2.resize(pred, original_shape[::-1])
    return pred


def split_image(gray, break_lines, margin):
    result = []
    chunk_width = gray.shape[1] // DEF_chunk_nums

    for idx, break_line in enumerate(break_lines):
        if idx == len(break_lines) - 1:
            continue
        tmp = gray.copy()
        for i, (above, below) in enumerate(zip(break_line, break_lines[idx + 1])):
            # handling edge case
            left_bound = i * chunk_width
            right_bound = (i + 1) * chunk_width if i != DEF_chunk_nums - 1 else gray.shape[1]
            tmp[:above, left_bound : right_bound] = 0
            tmp[below:, left_bound : right_bound] = 0
        sub_img = tmp[min(break_line) - margin : max(break_lines[idx + 1]) + margin, :]
        result.append(sub_img)

    return result


def draw_line_on_image(color, break_lines):
    chunk_width = color.shape[1] // DEF_chunk_nums
    for idx, break_line in enumerate(break_lines):
        for i, line in enumerate(break_line):
            left_bound = i * chunk_width
            right_bound = (i + 1) * chunk_width if i != DEF_chunk_nums - 1 else color.shape[1]
            color[line, left_bound : right_bound, :] = np.array([0, 0, 255])
            if i != 0 and previous != line:
                color[min(previous, line) : max(previous, line), i * chunk_width, :] = np.array([0, 0, 255])
            previous = line

    return color

def remove_background_imageset_by_opencv(imageset):
    # get histogram
    hist = np.histogram(imageset, bins=2)

    hists = list(zip(hist[0], hist[1]))
    # sort by frequency
    hist_sort_by_frequency = sorted(hists, key=lambda hist: hist[0])
    # most pixel color as background color
    main_color = hist_sort_by_frequency[1][1]
    # second most pixel color as pen color
    pen_color = hist_sort_by_frequency[0][1]

    if main_color > pen_color:
        # light background
        binary_img = cv2.adaptiveThreshold(imageset, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
        output_image = 255 - np.ma.array(imageset, mask = binary_img).filled(255)
    else:
        #dark background
        binary_img = cv2.adaptiveThreshold(imageset, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5)
        output_image = np.ma.array(imageset, mask = binary_img).filled(0)

    return output_image

if __name__ == '__main__':
    main()