import argparse
import os
import torch
import cv2
import numpy as np


import math
from glob import glob
from experiment import Structure, Experiment
from concern.config import Configurable, Config
from tqdm import tqdm
from scipy.signal import find_peaks

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
    parser.add_argument('--cuda', action='store_true', default=False, help='using cuda if called, otherwise cpu')
    
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    
    conf = Config()
    experiment_args = conf.compile(conf.load(args['xml']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    if args['cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # load model
    model = experiment.structure.builder.build(device)
    states = torch.load(
        args['resume'], map_location=device)

    model.load_state_dict(states, strict=False)
    model.eval()

    if os.path.isdir(args['image_path']):
        img_paths = [it for it in glob(f"{args['image_path']}/*.*") if "jpg" in it or "png" in it]
    else:
        img_paths = [args['image_path']]
    os.makedirs(args['result_dir'], exist_ok=True)
    for path in tqdm(img_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = remove_background_imageset_by_opencv(gray)
        heat = CNN_forward(model, img.astype('float32'), args['image_short_side'])
        all_boxes = split_line_with_contour(gray, heat, args['box_thresh'])
        for box in all_boxes:
            cv2.polylines(img, [box], True, (0, 255, 0), 1)
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args['result_dir'], os.path.basename(path)), np.hstack([heat, img]))
    
def split_line_with_contour(img, heat, thresh):
    if np.max(heat) > thresh:
        heat = cv2.equalizeHist((heat * 255).astype(np.uint8)).astype(np.float32) / 255
    binary = heat > thresh
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # determine how should we expand the contour to make them intersect
    iterations = 1
    kernel = np.array([[0, 1, 0]] * 3, dtype=np.uint8) # dilate on vertical only
    dilated = cv2.dilate((binary * 255).astype(np.uint8), kernel)
    contours_tmp, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    while len(contours_tmp) == len(contours):
        dilated = cv2.dilate(dilated, kernel)
        contours_tmp, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        iterations += 1
        if iterations == min(img.shape[:2]):
            break
    
    # expand the contour right just before they intersect to each other
    iterations -= 1
    dilated = cv2.dilate((binary * 255).astype(np.uint8), np.ones((3,3)), iterations=iterations)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # hoge = np.dstack([img.copy()] * 3)
    # cv2.drawContours(hoge, contours_tmp, -1, (0, 255, 0), 1)
    # cv2.imwrite("hoge.png", hoge)
    # exit()

    all_boxes = []
    for contour in contours:
        # straight rectangle
        x, y, w, h = cv2.boundingRect(contour)
        # piece = heat[y:y+h, x:x+w]
        if min(w, h) < 5:
            continue
        
        # skew rectangle
        min_rect = cv2.minAreaRect(contour)

        # approximated contour
        epsilon = 0.0002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx = approx.reshape((-1, 2))
        if approx.shape[0] >= 4:
            all_boxes.append(approx)
        else:
            all_boxes.append(min_rect)

    return all_boxes

        
        


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