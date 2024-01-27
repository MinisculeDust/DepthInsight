from getPatches_cv2 import extractTexture
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--patch_num", type=int, help="set patch size", default=464)
parser.add_argument("--training_dataset", type=str, help="Training Dataset path", default='')
parser.add_argument("--testing_dataset", type=str, help="Testing Dataset path", default='')
args = parser.parse_args()

patch_num = args.patch_num

testing_data_path = args.testing_dataset
training_data_path = args.training_dataset

with open(training_data_path, 'r') as f:
    filenames = f.readlines()

with open(testing_data_path, 'r') as f:
    filenames_testing = f.readlines()

replacedKeyword = '/NYU_large_texture_' + str(patch_num) + '/'

for i, item in enumerate(filenames_testing):
    extractTexture(filenames_testing[i].split(' ')[0], filenames_testing[i].split(' ')[1], patch_num=patch_num,
                   sourceKeyword='/NYU_large/', replacedKeyword=replacedKeyword, save_matrix=True, grayscale=True)

for i, item in enumerate(filenames):
    extractTexture(filenames[i].split(' ')[0], filenames[i].split(' ')[1], patch_num=patch_num, sourceKeyword='/NYU_large/', replacedKeyword=replacedKeyword, save_matrix=False, grayscale=True)