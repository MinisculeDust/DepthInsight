import random
import numpy as np
import getPatches_cv2
import json
import cv2
import matplotlib.pyplot as plt



img_path = 'texture_sample.png' # original image
depth_path = 'texture_output.png'
recover_path = 'texture_matrix.json' # path of the json file that stores the recover list
gt_depth_path = 'texture_gt.png'
PATCH_NUM = 16

depth_path = depth_path.replace('\n', '')
depth_BGR = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
depth_gt_path = gt_depth_path.replace('\n', '')
depth_gt_BGR = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
# depth_gt_RGB = cv2.cvtColor(depth_gt_BGR, cv2.COLOR_BGR2RGB)

image_path = img_path.replace('\n', '')
image_BGR = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if len(depth_BGR.shape) == 3 and depth_BGR.shape[2] == 4:
    depth = depth_BGR[:, :, :3]
    depth_gt = depth_gt_BGR[:, :, :3]
    image = image_BGR
    height, width, _ = depth.shape
else:
    depth = depth_BGR
    depth_gt = depth_gt_BGR
    image = image_BGR
    height, width = depth.shape
PADDING = 0

IMAGE_ROW = int(height / PATCH_NUM)
IMAGE_COLUMN = int(width / PATCH_NUM)
depth_list_shuffled = getPatches_cv2.cut_image(depth, PATCH_NUM, m=IMAGE_ROW, n=IMAGE_COLUMN)
depth_gt_list_shuffled = getPatches_cv2.cut_image(depth_gt, PATCH_NUM, m=IMAGE_ROW, n=IMAGE_COLUMN)
image_list_shuffled = getPatches_cv2.cut_image(image, PATCH_NUM, m=IMAGE_ROW, n=IMAGE_COLUMN)

recover_list = list(range(1, len(depth_list_shuffled) + 1))

with open(recover_path, 'r') as json_file:
    recover_list_shuffled = json.load(json_file)

depth_list_restored = [depth_list_shuffled[recover_list_shuffled.index(i)] for i in recover_list]
depth_gt_list_restored = [depth_gt_list_shuffled[recover_list_shuffled.index(i)] for i in recover_list]
image_list_restored = [image_list_shuffled[recover_list_shuffled.index(i)] for i in recover_list]

IMAGE_SIZE_WIDTH = PATCH_NUM
IMAGE_SIZE_HEIGHT = PATCH_NUM
IMAGE_ROW = int(height / PATCH_NUM)
IMAGE_COLUMN = int(width / PATCH_NUM)

depth_composed_shuffled = getPatches_cv2.image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, depth_list_shuffled)
depth_composed = getPatches_cv2.image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, depth_list_restored)
depth_gt_composed = getPatches_cv2.image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, depth_gt_list_restored)
image_composed = getPatches_cv2.image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, image_list_restored)

plt.imshow(depth_gt_composed/255.0)
plt.imsave('texture_gt_restored.png', depth_gt_composed/255.0, dpi=300)
plt.show()

plt.imshow(depth_composed/255.0)
plt.imsave('texture_output_restored.png', depth_composed/255.0, dpi=300)
plt.show()
