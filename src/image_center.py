import json
import math
import os
import re

import cv2
from tqdm import tqdm

label = {
    '061_foam_brick': 0,
    'green_basketball': 1,
    'salt_cylinder': 2,
    'shiny_toy_gun': 3,
    'stanley_screwdriver': 4,
    'strawberry': 5,
    'toothpaste_box': 6,
    'toy_elephant': 7,
    'whiteboard_spray': 8,
    'yellow_block': 9,
}


def pre_process(base_path):
    print(base_path)
    object_folders = os.listdir(base_path)
    object_info = {}
    object_list = []
    for object_folder in tqdm(object_folders):
        hit_segments = os.listdir(os.path.join(base_path, object_folder))
        direct_info = {}
        hit_segments.sort(key=int)
        for hit_segment in tqdm(hit_segments):
            mask_images = os.listdir(os.path.join(base_path, object_folder, hit_segment, 'mask'))
            mask_centers = []
            mask_images.sort()
            for mask_image in mask_images:
                mask = cv2.imread(os.path.join(base_path, object_folder, hit_segment, 'mask', mask_image))
                M = cv2.moments(mask[:, :, 0])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                mask_centers.append([cx, cy])
                # c22.imread -> y, x | BGR
                # mask[cy, cx, 2] = 255
                # mask[cy, cx, 1] = 0
                # mask[cy, cx, 0] = 0
                # cv2.imwrite('tmp.png', mask)
            angle = math.atan2(mask_centers[-1][1] - mask_centers[0][1], mask_centers[-1][0] - mask_centers[0][0]) * 180 / math.pi
            direct_info[hit_segment] = [mask_centers[0], mask_centers[-1], angle]
            direct_dict = {'class': label[object_folder], 'label': hit_segment, 'angle': angle}
            object_list.append(direct_dict)
        object_info[label[object_folder]] = direct_info

    return object_info, object_list


def test_mask_process(base_path):
    print(base_path)
    # object_info = {}
    object_list = []
    hit_segments = [file for file in sorted(os.listdir(base_path)) if not re.match(r".+\.(pkl|npy)", file)]
    # direct_info = {}
    for hit_segment in tqdm(hit_segments):
        mask_images = sorted(os.listdir(os.path.join(base_path, hit_segment, 'mask')))
        mask_centers = []
        for mask_image in mask_images:
            mask = cv2.imread(os.path.join(base_path, hit_segment, 'mask', mask_image))
            M = cv2.moments(mask[:, :, 0])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            mask_centers.append([cx, cy])
        angle = math.atan2(mask_centers[-1][1] - mask_centers[0][1], mask_centers[-1][0] - mask_centers[0][0]) * 180 / math.pi
        # direct_info[hit_segment] = [mask_centers[0], mask_centers[-1], angle]
        direct_dict = {'label': hit_segment, 'angle_mask': angle, 'pos_mask': mask_centers[-1]}
        object_list.append(direct_dict)
    # object_info[label[object_folder]] = direct_info

    return object_list
