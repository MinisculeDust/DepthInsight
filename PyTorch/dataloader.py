
from PIL import Image, ImageFile
import cv2
import os
import random
import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import DepthNorm
from torch.utils.data.dataloader import default_collate

print('loading the correct one 20230921--16--08')

def phaseScramble_depth(rgb_path, depth_path):

    img_BGR = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    img = img_BGR

    depth_BGR = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    # depth = cv2.merge((depth_BGR,depth_BGR,depth_BGR))
    depth = depth_BGR
    # rescale = 'off'
    p = 1

    # imclass = class(im); % get class of image

    # im = np.double(img)
    im = img
    imSize = img.shape
    depthSize = depth.shape

    # RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[1], imSize[2]))) # generate random phase structure in range p (between 0 and 1)
    RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[0], imSize[1]))) # generate random phase structure in range p (between 0 and 1)
    # RandomPhase(1) = 0 # leave out the DC value
    RandomPhase[0] = 0 # leave out the DC value

    if len(imSize) == 2:
        imSize[2] = 1

    # preallocate
    imFourier = np.zeros(imSize, dtype=complex)
    imFourier_depth = np.zeros(depthSize, dtype=complex)
    Amp = np.zeros(imSize)
    Amp_depth = np.zeros(depthSize)
    Phase = np.zeros(imSize)
    Phase_depth = np.zeros(depthSize)
    imScrambled = np.zeros(imSize, dtype=complex)
    imScrambled_depth = np.zeros(depthSize, dtype=complex)

    # for layer = 1:imSize(3)
    for layer in range(imSize[2]):
        imFourier[:,:,layer] = np.fft.fft2(im[:,:,layer])         # Fast-Fourier transform
        Amp[:,:,layer] = abs(imFourier[:,:,layer])         # amplitude spectrum
        Phase[:,:,layer] = np.angle(imFourier[:,:,layer])     # phase spectrum
        Phase[:,:,layer] = Phase[:,:,layer] + RandomPhase  # add random phase to original phase
        # combine Amp and Phase then perform inverse Fourier
        imScrambled[:,:,layer] = np.fft.ifft2(Amp[:,:,layer] * np.exp(np.sqrt(-1+0j)*(Phase[:,:,layer])))
    imScrambled = imScrambled.real # get rid of imaginery part in image (due to rounding error)

    imFourier_depth[:, :] = np.fft.fft2(depth[:, :])  # Fast-Fourier transform
    Amp_depth[:, :] = abs(imFourier_depth[:, :])  # amplitude spectrum
    Phase_depth[:, :] = np.angle(imFourier_depth[:, :])  # phase spectrum
    Phase_depth[:, :] = Phase_depth[:, :] + RandomPhase  # add random phase to original phase
    # combine Amp and Phase then perform inverse Fourier
    imScrambled_depth[:, :] = np.fft.ifft2(Amp_depth[:, :] * np.exp(np.sqrt(-1 + 0j) * (Phase_depth[:, :])))
    imScrambled_depth = imScrambled_depth.real # get rid of imaginery part in image (due to rounding error)

    return imScrambled.astype(np.float32), imScrambled_depth.astype(np.float32)

def phaseScramble_colour_grayscale(rgb_path, depth_path):

    img_BGR = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    depth_BGR = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    # depth = cv2.merge((depth_BGR,depth_BGR,depth_BGR))
    depth = depth_BGR
    # rescale = 'off'
    p = 1

    im = img
    imSize = img.shape
    depthSize = depth.shape

    # RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[1], imSize[2]))) # generate random phase structure in range p (between 0 and 1)
    RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[0], imSize[1]))) # generate random phase structure in range p (between 0 and 1)
    # RandomPhase(1) = 0 # leave out the DC value
    RandomPhase[0] = 0 # leave out the DC value ????

    # preallocate
    imFourier = np.zeros(imSize, dtype=complex)
    imFourier_depth = np.zeros(depthSize, dtype=complex)
    Amp = np.zeros(imSize)
    Amp_depth = np.zeros(depthSize)
    Phase = np.zeros(imSize)
    Phase_depth = np.zeros(depthSize)
    imScrambled = np.zeros(imSize, dtype=complex)
    imScrambled_depth = np.zeros(depthSize, dtype=complex)

    # for layer = 1:imSize(3)
    # for layer in range(imSize[2]):
    imFourier[:,:] = np.fft.fft2(im[:,:])         # Fast-Fourier transform
    Amp[:,:] = abs(imFourier[:,:])         # amplitude spectrum
    Phase[:,:] = np.angle(imFourier[:,:])     # phase spectrum
    Phase[:,:] = Phase[:,:] + RandomPhase  # add random phase to original phase
    # combine Amp and Phase then perform inverse Fourier
    imScrambled[:,:] = np.fft.ifft2(Amp[:,:] * np.exp(np.sqrt(-1+0j)*(Phase[:,:])))
    imScrambled = imScrambled.real # get rid of imaginery part in image (due to rounding error)

    imFourier_depth[:, :] = np.fft.fft2(depth[:, :])  # Fast-Fourier transform
    Amp_depth[:, :] = abs(imFourier_depth[:, :])  # amplitude spectrum
    Phase_depth[:, :] = np.angle(imFourier_depth[:, :])  # phase spectrum
    Phase_depth[:, :] = Phase_depth[:, :] + RandomPhase  # add random phase to original phase
    # combine Amp and Phase then perform inverse Fourier
    imScrambled_depth[:, :] = np.fft.ifft2(Amp_depth[:, :] * np.exp(np.sqrt(-1 + 0j) * (Phase_depth[:, :])))
    # for layer in range(imSize[2]):
    #     imFourier_depth[:,:,layer] = np.fft.fft2(depth[:,:,layer])         # Fast-Fourier transform
    #     Amp_depth[:,:,layer] = abs(imFourier_depth[:,:,layer])         # amplitude spectrum
    #     Phase_depth[:,:,layer] = np.angle(imFourier_depth[:,:,layer])     # phase spectrum
    #     Phase_depth[:,:,layer] = Phase_depth[:,:,layer] + RandomPhase  # add random phase to original phase
    #     # combine Amp and Phase then perform inverse Fourier
    #     imScrambled_depth[:,:,layer] = np.fft.ifft2(Amp_depth[:,:,layer] * np.exp(np.sqrt(-1+0j)*(Phase_depth[:,:,layer])))
    imScrambled_depth = imScrambled_depth.real # get rid of imaginery part in image (due to rounding error)


    # rgb_saved_path = saving_root + 'nyu_images/' + rgb_path.split('/')[-1].split('.')[0] + '.jpg'
    # depth_saved_path = saving_root + 'nyu_depths/' +  depth_path.split('/')[-1].split('.')[0] + '.png'
    # cv2.imwrite(rgb_saved_path, imScrambled.astype(np.float32))
    # # cv2.imwrite(depth_saved_path, imScrambled_depth.astype(np.float32)[:, :, 0])
    # cv2.imwrite(depth_saved_path, imScrambled_depth[:, :, 0].astype(np.float32))

    # return imScrambled.astype(np.float32), imScrambled_depth[:, :, 0].astype(np.float32)
    return imScrambled.astype(np.float32), imScrambled_depth.astype(np.float32)

def phaseScramble_saturation(rgb_path, depth_path):

    img_BGR = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    img_hsv = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)  # separate three channels
    # img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    img = S

    depth_BGR = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    # depth = cv2.merge((depth_BGR,depth_BGR,depth_BGR))
    depth = depth_BGR
    # rescale = 'off'
    p = 1

    # imclass = class(im); % get class of image

    # im = np.double(img)
    im = img
    imSize = img.shape
    depthSize = depth.shape

    # RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[1], imSize[2]))) # generate random phase structure in range p (between 0 and 1)
    RandomPhase = p * np.angle(np.fft.fft2(np.random.rand(imSize[0], imSize[1]))) # generate random phase structure in range p (between 0 and 1)
    # RandomPhase(1) = 0 # leave out the DC value
    RandomPhase[0] = 0 # leave out the DC value ????

    # preallocate
    imFourier = np.zeros(imSize, dtype=complex)
    imFourier_depth = np.zeros(depthSize, dtype=complex)
    Amp = np.zeros(imSize)
    Amp_depth = np.zeros(depthSize)
    Phase = np.zeros(imSize)
    Phase_depth = np.zeros(depthSize)
    imScrambled = np.zeros(imSize, dtype=complex)
    imScrambled_depth = np.zeros(depthSize, dtype=complex)

    # for layer = 1:imSize(3)
    # for layer in range(imSize[2]):
    imFourier[:,:] = np.fft.fft2(im[:,:])         # Fast-Fourier transform
    Amp[:,:] = abs(imFourier[:,:])         # amplitude spectrum
    Phase[:,:] = np.angle(imFourier[:,:])     # phase spectrum
    Phase[:,:] = Phase[:,:] + RandomPhase  # add random phase to original phase
    # combine Amp and Phase then perform inverse Fourier
    imScrambled[:,:] = np.fft.ifft2(Amp[:,:] * np.exp(np.sqrt(-1+0j)*(Phase[:,:])))
    imScrambled = imScrambled.real # get rid of imaginery part in image (due to rounding error)

    imFourier_depth[:, :] = np.fft.fft2(depth[:, :])  # Fast-Fourier transform
    Amp_depth[:, :] = abs(imFourier_depth[:, :])  # amplitude spectrum
    Phase_depth[:, :] = np.angle(imFourier_depth[:, :])  # phase spectrum
    Phase_depth[:, :] = Phase_depth[:, :] + RandomPhase  # add random phase to original phase
    # combine Amp and Phase then perform inverse Fourier
    imScrambled_depth[:, :] = np.fft.ifft2(Amp_depth[:, :] * np.exp(np.sqrt(-1 + 0j) * (Phase_depth[:, :])))
    imScrambled_depth = imScrambled_depth.real # get rid of imaginery part in image (due to rounding error)

    def hue_range(hue): # for H maps
        hue *= 2.0
        hue = np.where(hue < 0, (hue + 360) % 360, hue)
        hue = np.where(hue > 360, hue % 360, hue)
        return hue / 2.0

    return imScrambled.astype(np.float32), imScrambled_depth.astype(np.float32)


def convert_to_edges(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Sobel operator to detect edges
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    return edges

def convert_to_edges_Canny(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    return edges

def image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, padding, IMAGES_LIST):

    if len(IMAGES_LIST[0].shape) < 3:
        to_image = np.zeros((IMAGE_ROW * IMAGE_SIZE_HEIGHT + padding * (IMAGE_ROW-1), IMAGE_COLUMN * IMAGE_SIZE_WIDTH + padding * (IMAGE_COLUMN-1)))
    else:
        to_image = np.zeros((IMAGE_ROW * IMAGE_SIZE_HEIGHT + padding * (IMAGE_ROW-1), IMAGE_COLUMN * IMAGE_SIZE_WIDTH + padding * (IMAGE_COLUMN-1), 3))
        # to_image = np.ones((IMAGE_ROW * IMAGE_SIZE_HEIGHT + padding * (IMAGE_ROW-1), IMAGE_COLUMN * IMAGE_SIZE_WIDTH + padding * (IMAGE_COLUMN-1), 3)) * 255 # double-check black blocks coming from

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
    # print(box_list)
    # image_list = [image.crop(box) for box in box_list]  #Image.crop(left, up, right, below)
    image_list = []
    for box in box_list:
        # image_list.append(image[box[1]:box[3], box[0]:box[2]])
        image_list.append(image[box[0]:box[2], box[1]:box[3]])
    # image_list = [image.crop(box) for box in box_list]  #Image.crop(left, up, right, below)
    return image_list
def extractTexture(rgb_path, depth_path, patch_size):

    img_BGR = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    image = img_BGR

    depth_BGR = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # depth = cv2.cvtColor(depth_BGR, cv2.COLOR_BGR2RGB)
    depth = depth_BGR

    PATCH_NUM = patch_size  # (624/16) * (464/16)

    PADDING = 0
    if len(image.shape) < 3:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    # IMAGE_SIZE_WIDTH = int(width / PATCH_NUM)
    # IMAGE_SIZE_HEIGHT = int(height / PATCH_NUM)
    # IMAGE_ROW = PATCH_NUM
    # IMAGE_COLUMN = PATCH_NUM * int(max(width, height) / min(width, height))
    IMAGE_SIZE_WIDTH = PATCH_NUM
    IMAGE_SIZE_HEIGHT = PATCH_NUM
    IMAGE_ROW = int(height / PATCH_NUM)
    IMAGE_COLUMN = int(width / PATCH_NUM)

    image_list = cut_image(image, patch_num=PATCH_NUM, m=IMAGE_ROW, n=IMAGE_COLUMN)
    depth_list = cut_image(depth, patch_num=PATCH_NUM, m=IMAGE_ROW, n=IMAGE_COLUMN)

    # zip RGB & depth paris, and then shuffle them
    zip_list = list(zip(image_list, depth_list))
    random.shuffle(zip_list)
    image_list_shuffled, depth_list_shuffled = zip(*zip_list)

    # save_images(IMAGES_LIST, save_path)


    rgb_composed_shuffled = image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, image_list_shuffled)
    depth_composed_shuffled = image_compose(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_ROW, IMAGE_COLUMN, PADDING, depth_list_shuffled)

    rgb_img = rgb_composed_shuffled.astype(np.uint8)
    depth_img = depth_composed_shuffled.astype(np.uint8)

    texture_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # return rgb_img, depth_img
    return texture_img, depth_img


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


def my_collate_fn(batch):
    #  fileter NoneType data
    batch = list(filter(lambda x:x['depth'] is not None and x['image'] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)

class DepthDataLoader(object):
    def __init__(self, args, mode, feature=''):
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.53607797, 0.53617338, 0.53618207], std=[0.31895092, 0.31896688, 0.31896867]) # for SUNCG training dataset
            # transforms.Normalize(mean=[0.48013042, 0.41071221, 0.39187948], std=[0.28875214, 0.29518897, 0.30795045]) # for NYU training dataset
        ]
        )
        transform_data_single = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        )

        if mode == 'train':
            # self.training_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            self.training_samples = DataLoadPreprocess(args, mode, feature, transform=transform_data)
            self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.bs,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'target':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, args.bs,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, feature, transform=transform_data)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('error')

class DepthDataLoader_evaluate_contrast(object):
    def __init__(self, args, mode, feature='', contrast=1):
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.53607797, 0.53617338, 0.53618207], std=[0.31895092, 0.31896688, 0.31896867]) # for SUNCG training dataset
            # transforms.Normalize(mean=[0.48013042, 0.41071221, 0.39187948], std=[0.28875214, 0.29518897, 0.30795045]) # for NYU training dataset
        ]
        )
        transform_data_single = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        )

        if mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess_evaluate_contrast(args, mode, feature, transform=transform_data, contrast=contrast)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('error')

class DepthDataLoader_evaluate_saturation(object):
    def __init__(self, args, mode, feature='', saturation=1):
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.53607797, 0.53617338, 0.53618207], std=[0.31895092, 0.31896688, 0.31896867]) # for SUNCG training dataset
            # transforms.Normalize(mean=[0.48013042, 0.41071221, 0.39187948], std=[0.28875214, 0.29518897, 0.30795045]) # for NYU training dataset
        ]
        )
        transform_data_single = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        )

        if mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess_evaluate_saturation(args, mode, feature, transform=transform_data, saturation=saturation)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('error')


class DepthDataLoader_evaluate(object):
    def __init__(self, args, mode):
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.53607797, 0.53617338, 0.53618207], std=[0.31895092, 0.31896688, 0.31896867]) # for SUNCG training dataset
            # transforms.Normalize(mean=[0.48013042, 0.41071221, 0.39187948], std=[0.28875214, 0.29518897, 0.30795045]) # for NYU training dataset
        ]
        )
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'target':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            if args.distributed:  # redundant. here only for readability and to be more explicit
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_data)
            if args.distributed:  # redundant. here only for readability and to be more explicit
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, feature='', transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'target' or mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.feature = feature
        # self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])

        if self.mode == 'train':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(self.args.data_path, sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            if self.feature == 'colour':
                image, depth_gt = phaseScramble_depth(rgb_path=image_path, depth_path=depth_path)

                # if image.max() > 1:
                #     image /= 255

                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'colour_grayscale':
                image, depth_gt = phaseScramble_colour_grayscale(rgb_path=image_path, depth_path=depth_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # if image.max() > 1:
                #     image /= 255

                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'saturation':
                image, depth_gt = phaseScramble_saturation(rgb_path=image_path, depth_path=depth_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # if image.max() > 1:
                #     image /= 255

                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'texture':
                image, depth_gt = extractTexture(rgb_path=image_path, depth_path=depth_path, patch_size=self.args.patchSize)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # if image.max() > 1:
                #     image = image.astype(np.float32)
                #     image /= 255.0
                depth_gt = depth_gt.astype('float32')
                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'shape':
                image = convert_to_edges(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            elif self.feature == 'shape2':
                image = convert_to_edges_Canny(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            elif self.feature == 'single_grayscale':
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # transform to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # transform to 3 channels
                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            elif self.feature == 'pseudo_rgb':
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            # elif self.feature == 'grayscale':
            #     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #     # for exr file, 3 channels are the same
            #     if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
            #         # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
            #         depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
            #         depth_gt = depth_gt.astype('float32')
            #         if depth_gt.max() > 1:
            #             depth_gt /= 255
            #         depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
            #                              doNorm=False)
            #     else:
            #         depth_gt = None
            else:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval, doNorm=False)
                else:
                    depth_gt = None

            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            sample = {'image': image, 'depth': depth_gt}

        # if self.mode == 'target':
        #     if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
        #         image_path = os.path.join(self.args.data_path, remove_leading_slash(sample_path.split()[3]))
        #         depth_path = os.path.join(self.args.gt_path, remove_leading_slash(sample_path.split()[4]))
        #     else:
        #         image_path = os.path.join(self.args.data_path, sample_path.split()[0])
        #         depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])
        #
        #     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #     if image is not None:
        #         if self.transform is not None:
        #             image = self.transform(image)
        #         else:
        #             image = image.transpose(2, 0, 1)
        #     else:
        #         image = None
        #
        #     # for exr file, 3 channels are the same
        #     if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
        #         depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
        #         depth_gt = depth_gt.astype('float32')
        #         if depth_gt.max() > 1:
        #             depth_gt /= 255
        #         depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval, doNorm=False)
        #     else:
        #         depth_gt = None
        #
        #     sample = {'image': image, 'depth': depth_gt}

        if self.mode == 'online_eval':
            image_path = os.path.join(self.args.data_path, sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])
            if self.feature == 'colour':
                image, depth_gt = phaseScramble_depth(rgb_path=image_path, depth_path=depth_path)

                # if image.max() > 1:
                #     image /= 255

                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'colour_grayscale':
                image, depth_gt = phaseScramble_colour_grayscale(rgb_path=image_path, depth_path=depth_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # if image.max() > 1:
                #     image /= 255

                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'saturation':
                image, depth_gt = phaseScramble_saturation(rgb_path=image_path, depth_path=depth_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # if image.max() > 1:
                #     image /= 255

                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'texture':
                image, depth_gt = extractTexture(rgb_path=image_path, depth_path=depth_path, patch_size=self.args.patchSize)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # if image.max() > 1:
                #     image = image.astype(np.float32)
                #     image /= 255.0
                depth_gt = depth_gt.astype('float32')
                if depth_gt.max() > 1:
                    depth_gt /= 255
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            elif self.feature == 'shape':
                image = convert_to_edges(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            elif self.feature == 'shape2':
                image = convert_to_edges_Canny(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            elif self.feature == 'single_grayscale':
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # transform to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # transform to 3 channels
                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            elif self.feature == 'pseudo_rgb':
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # transform to 3 channels
                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None
            else:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                # for exr file, 3 channels are the same
                if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
                    depth_gt = depth_gt.astype('float32')
                    if depth_gt.max() > 1:
                        depth_gt /= 255
                    depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                         doNorm=False)
                else:
                    depth_gt = None

            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            sample = {'image': image, 'depth': depth_gt, 'img_path': sample_path}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)

def adjust_contrast(image, contrast_factor):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray_image)

    # Create the degenerate (pure gray) image
    degenerate = np.full(image.shape, mean, dtype=np.uint8)

    # Blend the original image and the degenerate image
    enhanced_image = cv2.addWeighted(image, contrast_factor, degenerate, 1 - contrast_factor, 0)

    return enhanced_image

def change_image_saturation(image, saturation_factor):

    # 将图像从BGR转换为HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 调整饱和度
    image = image.astype('float64')  # 将图像转换为float，避免溢出
    image[:,:,1] = image[:,:,1]*saturation_factor  # 调整饱和度

    # 处理可能出现的溢出情况
    image[:,:,1][image[:,:,1]>255] = 255
    image = image.astype('uint8')  # 将图像转换回uint8

    # 将图像从HSV转换回BGR
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

class DataLoadPreprocess_evaluate_contrast(Dataset):
    def __init__(self, args, mode, feature='', transform=None, contrast=1, is_for_online_eval=False):
        self.args = args
        if mode == 'target' or mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.feature = feature
        self.contrast = contrast
        # self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])

        if self.mode == 'online_eval':
            image_path = os.path.join(self.args.data_path, sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            image = adjust_contrast(image, self.contrast)

        if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
            # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
            depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
            depth_gt = depth_gt.astype('float32')
            if depth_gt.max() > 1:
                depth_gt /= 255
            depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                 doNorm=False)
        else:
            depth_gt = None

        if image is not None:
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.transpose(2, 0, 1)
        else:
            image = None

        sample = {'image': image, 'depth': depth_gt}
            # np.transpose(image.cpu().numpy(), (1, 2, 0))
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)



class DataLoadPreprocess_evaluate_saturation(Dataset):
    def __init__(self, args, mode, feature='', transform=None, saturation=1, is_for_online_eval=False):
        self.args = args
        if mode == 'target' or mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.feature = feature
        self.saturation = saturation
        # self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])

        if self.mode == 'online_eval':
            image_path = os.path.join(self.args.data_path, sample_path.split()[0])
            depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            image = change_image_saturation(image, self.saturation)

        if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
            # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
            depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # NYU exr shape ==> (464, 624)
            depth_gt = depth_gt.astype('float32')
            if depth_gt.max() > 1:
                depth_gt /= 255
            depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                 doNorm=False)
        else:
            depth_gt = None

        if image is not None:
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.transpose(2, 0, 1)
        else:
            image = None

        sample = {'image': image, 'depth': depth_gt}
            # np.transpose(image.cpu().numpy(), (1, 2, 0))
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)

