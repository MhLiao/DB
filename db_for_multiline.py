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
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype('float32')
        heat = CNN_forward(model, img, args['image_short_side'])
        result = run(img, heat, output_path=None)
        cv2.imwrite(os.path.join(args['result_dir'], os.path.basename(path)), result)
    

def run(image, heat_map, output_path, margin=0):
    # image, heat_map = preprocess(image, heat_map)
    # main process
    # calculate histogram in horizontal direction
    h, w = heat_map.shape[:2]

    histogram = np.sum(heat_map, axis=1)

    # estimate text height (in pixel)
    rough_text_height = estimate_text_height_2(heat_map)

    # generate straightline
    valleys, _ = find_peaks(-histogram, distance=rough_text_height * 0.8)
    peaks, _ = find_peaks(histogram, distance=rough_text_height * 0.8)

    straight_lines = valleys

    # double check
    for v1, v2 in zip(valleys, valleys[1:]):
        peaks_inside = [peak for peak in peaks if (v1 < peak and peak < v2)]
        if len(peaks_inside) < 1: # there exists 1 wrong valley
            wrong_valley = v1 if histogram[v1] > histogram[v2] else v2
            straight_lines = straight_lines[straight_lines != wrong_valley]
        #TODO do something in case there exists more than 2 peaks

    straight_lines = list(straight_lines)
    if len(straight_lines) == 0:
        straight_lines = [0, heat_map.shape[0] - 1]
    if straight_lines[0] > rough_text_height:
        straight_lines.insert(0, 0)
    if heat_map.shape[0] - 1 - straight_lines[-1] > rough_text_height:
        straight_lines.append(heat_map.shape[0] - 1)
    if DEBUG:
        tmp = image.copy()
        for line in straight_lines:
            tmp[line,:,0] = 0
            tmp[line,:,1] = 0
            tmp[line,:,2] = 255
        cv2.imwrite("DEBUG/1_straight.png", tmp)
    break_lines = line_adjustment(heat_map, straight_lines, text_height=int(rough_text_height), score_text=heat_map)

    if DEBUG:
        result = draw_line_on_image(image.copy(), break_lines)
        tmp1 = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # tmp1 = np.dstack([heat_map * 255, heat_map * 255, heat_map * 255])
        result = np.hstack([tmp1, result])
        return result
    else:
        result = split_image(image, break_lines, margin)
        image_lines = []
        for i, item in enumerate(result):
            img_path = os.path.splitext(output_path)[0] + "_" + str(i) + ".png"
            image_lines.append((item, img_path))
        return image_lines

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

def estimate_text_height_2(score_text, binary_thresh=0.5):
    img = np.zeros(score_text.shape, np.uint8)
    img[score_text > binary_thresh] = 255
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))
    box_heights = []
    for c in contours:
        poly = cv2.approxPolyDP(c, 1, True)
        rect = cv2.boundingRect(poly)
        box_heights.append(rect[3])
    if len(box_heights) == 0 or np.median(box_heights) < 5: # handle rare case
        return 5
    return np.median(box_heights)

def line_adjustment(gray, lines, text_height, score_text):
    chunk_width = gray.shape[1] // DEF_chunk_nums
    histograms = []
    histograms2 = []
    break_lines = []
    for i in range(DEF_chunk_nums):
        chunk_img = gray[:, chunk_width * i : chunk_width * (i + 1)]
        chunk_img = cv2.medianBlur(chunk_img, 5)
        histograms.append(np.sum(chunk_img, axis=1) / 255)
        chunk_st = score_text[:, chunk_width * i : chunk_width * (i + 1)]
        histograms2.append(np.sum(chunk_st, axis=1))
    for idx, line in enumerate(lines):
        if idx == 0:  # no adjust for first line
            break_lines.append([0] * DEF_chunk_nums)
            continue
        elif idx == len(lines) - 1:  # no adjust for last line
            break_lines.append([gray.shape[0] - 1] * DEF_chunk_nums)
            continue
        center_line = line
        break_line = []
        for chunk_idx in range(DEF_chunk_nums):
            histogram = histograms[chunk_idx]
            histogram2 = histograms2[chunk_idx]
            upper_line = break_lines[idx - 1][chunk_idx]
            below_line = lines[idx + 1]

            # change coordinate
            histogram = histogram[upper_line:below_line]
            histogram2 = histogram2[upper_line:below_line]
            center_line -= upper_line

            peaks, _ = find_peaks(histogram2, distance=text_height * 0.8, height=histogram2.max() / 2)

            if len(peaks) < 2:
                tmp = center_line if chunk_idx == 0 else break_line[chunk_idx - 1] - upper_line
                start = max(0, tmp - text_height // 2)
                end = min(histogram.shape[0], tmp + text_height // 2)
            else:
                start = peaks[0]
                end = peaks[-1]
            if chunk_idx == 0:
                cost = calculate_cost(histogram[start:end], center_line - start, w1=1.0, w2=0, w3=0.001) # deactivate cummulation cost
            else:
                cost = calculate_cost(histogram[start:end], center_line - start)
            # return minimum position as global coordinate
            minimum_position = np.argmin(cost) + start + upper_line
            # update break line
            break_line.append(minimum_position)
            center_line = minimum_position


        break_lines.append(break_line)
    return break_lines

def calculate_cost(histogram, start_point, w1=1.0, w2=0.01, w3=0.001):
    """
    Calculate cost to find local minimum in adjust_line
    Args:
        histogram: histogram surrounding start point
        start_point: current position of break_line
    Returns:
        cost: 1D numpy array of cost value
    """
    if histogram.shape[0] == 0: # the step of determining histogram failed
        return np.array([0]) # return dummy array to catch the unusual case
    cost = np.zeros((histogram.shape))
    for i in range(histogram.shape[0]):
        cost1 = histogram[i]  # histogram difference
        range_1, range_2 = min(start_point, i), max(start_point, i)
        cost2 = np.sum(
            histogram[range_1 : range_2 + 1]
        )  # sum of histogram from start point to i
        cost3 = abs(i - start_point)  # histogram distance
        cost[i] = cost1 * w1 + cost2 * w2 + cost3 * w3
    return cost

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

if __name__ == '__main__':
    main()