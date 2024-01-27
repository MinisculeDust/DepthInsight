import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import json

saved_root_path = '/Users/yihongwu/Documents/PyCharm/ViT/DepthInsight/trained_models/DepthInsight/DI_outputs/'

phased_scrambled_sample_path = ''
phased_scrambled_sample_depth_path = ''
phased_scrambled_output_depth_path = ''
randomMatrix_path = ''

# imScrambled = cv2.imread(phased_scrambled_sample_path, cv2.IMREAD_UNCHANGED)
# imScrambled = cv2.imread(phased_scrambled_sample_depth_path, cv2.IMREAD_GRAYSCALE)
imScrambled = cv2.imread(phased_scrambled_output_depth_path, cv2.IMREAD_GRAYSCALE)

'''
add noise to the whole image
'''
if False:
    # Gaussian Noise
    mean = 0
    stddev = 25
    noise = np.random.normal(mean, stddev, imScrambled.shape).astype(np.uint8)
    # add noise to the image
    noisy_image = cv2.add(imScrambled, noise)
    imScrambled = noisy_image
'''
add noise to the central region
'''
if False:
    # get the size of the image
    height, width, _ = imScrambled.shape
    # coordinate of the top left corner
    top_left = (width // 4, height // 4)  # 左上角坐标
    bottom_right = (3 * width // 4, 3 * height // 4)  # 右下角坐标
    # Gaussian Noise
    mean = 0
    stddev = 25
    # 为你要添加噪声的区域创建一个掩码
    mask = np.zeros_like(imScrambled[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])

    # 生成与掩码相同形状的噪声
    noise = np.random.normal(mean, stddev, mask.shape).astype(np.uint8)

    # 将噪声添加到感兴趣的区域
    imScrambled[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cv2.add(
        imScrambled[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], noise, dtype=cv2.CV_8U)

# plt.imshow(imScrambled/255.0)
# plt.imsave(root_path + '/color_sample_noised.png', np.clip(imScrambled/255.0, 0, 1), dpi=300)
# plt.show()

RandomPhase = np.load(randomMatrix_path)

imSize = imScrambled.shape
Amp = np.zeros(imSize)
Phase = np.zeros(imSize)
imFourier = np.zeros(imSize, dtype=complex)
reconstructed_im = np.zeros(imSize, dtype=complex)

if len(imSize) == 2:
    imFourier[:, :] = np.fft.fft2(imScrambled[:, :])
    Amp[:, :] = np.abs(imFourier[:, :])
    Phase[:, :] = np.angle(imFourier[:, :]) - RandomPhase

    # 逆操作
    reconstructed_phase = np.exp(1j * Phase[:, :])
    reconstructed_fourier = Amp[:, :] * reconstructed_phase

    # 逆FFT
    reconstructed_im[:, :] = np.fft.ifft2(reconstructed_fourier)

    reconstructed_im = reconstructed_im.real
else:
    for layer in range(imSize[2]):
        # FFT变换
        imFourier[:, :, layer] = np.fft.fft2(imScrambled[:, :, layer])
        Amp[:, :, layer] = np.abs(imFourier[:, :, layer])
        Phase[:, :, layer] = np.angle(imFourier[:, :, layer]) - RandomPhase

        # 逆操作
        reconstructed_phase = np.exp(1j * Phase[:, :, layer])
        reconstructed_fourier = Amp[:, :, layer] * reconstructed_phase

        # 逆FFT
        reconstructed_im[:, :, layer] = np.fft.ifft2(reconstructed_fourier)

    # 取实部
    reconstructed_im = reconstructed_im.real

plt.imshow(reconstructed_im/255.0)
# plt.imsave(root_path + '/restored_saturation.png', np.clip(reconstructed_im/255.0, 0, 1), dpi=300)
plt.show()