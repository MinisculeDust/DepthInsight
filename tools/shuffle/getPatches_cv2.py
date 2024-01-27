
import os.path
from PIL import Image
import sys
import torchvision.transforms as transforms
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

def image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, padding, IMAGES_LIST):


    if len(IMAGES_LIST[0].shape) < 3:
        to_image = np.zeros((IMAGE_ROW * IMAGE_SIZE_HEIGHT + padding * (IMAGE_ROW-1), IMAGE_COLUMN * IMAGE_SIZE_WIDTH + padding * (IMAGE_COLUMN-1)))
    else:
        to_image = np.zeros((IMAGE_ROW * IMAGE_SIZE_HEIGHT + padding * (IMAGE_ROW-1), IMAGE_COLUMN * IMAGE_SIZE_WIDTH + padding * (IMAGE_COLUMN-1), 3))

    img_num = 0
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = IMAGES_LIST[img_num]

            to_image[
            (y - 1) * IMAGE_SIZE_HEIGHT + padding * (y - 1):(y - 1) * IMAGE_SIZE_HEIGHT + padding * (y - 1) + from_image.shape[0],
            (x - 1) * IMAGE_SIZE_WIDTH + padding * (x - 1):(x - 1) * IMAGE_SIZE_WIDTH + padding * (x - 1) + from_image.shape[1]] = from_image
            img_num += 1
    return to_image


def cut_image(image, patch_num, m, n):
    box_list = []
    for i in range(0, m * patch_num, patch_num):
        for j in range(0, n * patch_num, patch_num):
            # patch = image[i:i + patch_num, j:j + patch_num]
            box_list.append((i, j, i+patch_num, j+patch_num))
    image_list = []
    for box in box_list:
        image_list.append(image[box[0]:box[2], box[1]:box[3]])
    return image_list


# 保存
def save_images(image_list, save_path):
    index = 1
    for image in image_list:
        image.save(os.path.join(save_path, str(index) + '.png'))
        index += 1


def extractTexture(rgb_path, depth_path, patch_num = 16, sourceKeyword='', replacedKeyword='', save_matrix=False, grayscale=True, saving_root='./'):

    img_BGR = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    image = img_BGR

    depth_path = depth_path.replace('\n', '')
    depth_BGR = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth_BGR

    PADDING = 0
    if len(image.shape) < 3:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    IMAGE_SIZE_WIDTH = patch_num
    IMAGE_SIZE_HEIGHT = patch_num
    IMAGE_ROW = int(height / patch_num)
    IMAGE_COLUMN = int(width / patch_num)

      # (624/16) * (464/16)

    image_list = cut_image(image, patch_num=patch_num, m=IMAGE_ROW, n=IMAGE_COLUMN)
    depth_list = cut_image(depth, patch_num=patch_num, m=IMAGE_ROW, n=IMAGE_COLUMN)
    recover_list = list(range(1, len(depth_list) + 1))

    # zip RGB & depth pairs, and then shuffle them
    zip_list = list(zip(image_list, depth_list, recover_list))
    random.shuffle(zip_list)
    image_list_shuffled, depth_list_shuffled, recover_list_shuffled = zip(*zip_list)


    rgb_composed_shuffled = image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, image_list_shuffled)
    depth_composed_shuffled = image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, depth_list_shuffled)

    rgb_saved_path = os.path.dirname(rgb_path).replace(sourceKeyword, replacedKeyword) + '/' + rgb_path.split('/')[-1].split('.')[0] + '.jpg'
    depth_saved_path = os.path.dirname(depth_path).replace(sourceKeyword, replacedKeyword) + '/' + depth_path.split('/')[-1].split('.')[0] + '.png'
    recover_saved_path = os.path.dirname(depth_path).replace(sourceKeyword, replacedKeyword) + '/' + depth_path.split('/')[-1].split('.')[0] + '.json'
    os.makedirs(os.path.dirname(rgb_saved_path), exist_ok=True)
    os.makedirs(os.path.dirname(depth_saved_path), exist_ok=True)
    os.makedirs(os.path.dirname(recover_saved_path), exist_ok=True)

    rgb_composed_shuffled = rgb_composed_shuffled.astype(np.uint8)
    if grayscale:
        rgb_composed_shuffled = cv2.cvtColor(rgb_composed_shuffled, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(rgb_saved_path, rgb_composed_shuffled)
    cv2.imwrite(depth_saved_path, depth_composed_shuffled.astype(np.uint8))
    if save_matrix:
        with open(recover_saved_path, 'w') as json_file:
            json.dump(recover_list_shuffled, json_file)