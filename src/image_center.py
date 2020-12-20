import json
import math
import os

import cv2
from tqdm import tqdm


def pre_process(base_path):
    print(base_path)
    object_folders = os.listdir(base_path)
    object_info = {}
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
        object_info[object_folder] = direct_info

    return object_info


def main():
    print('main')
    object_info = pre_process('dataset/train')
    json.dump(object_info, open('mask_processed.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    main()
