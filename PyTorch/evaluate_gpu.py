

from PIL import Image
import numpy as np
import time
import cv2
import torch
import csv
from sklearn.utils import shuffle
import os
import imageio
from matplotlib import pyplot as plt
from skimage.transform import resize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The Device is: ', str(device))

def loadCSV(file_name):
    f = open(file_name, 'r')
    csvreader = csv.reader(f)
    data = list(csvreader)

    # check and delete inexistent data
    for index, image in enumerate(data):
        if os.path.exists(image[0]) == False or os.path.exists(image[1]) == False:
            del data[index]

    data = shuffle(data, random_state=0)

    print('Loaded ({0}).'.format(len(data)))

    return data


def transfer_16bit_to_8bit_grey(image_path):
    image_16bit = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    # Not all the values are the same, but performance really similar from OpenHDR
    # image_16bit = image_16bit[:, :, -1]
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit

def transfer_16bit_to_float(image_path):
    image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Not all the values are the same, but performance really similar from OpenHDR
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_float = (image_16bit - min_16bit) / float((max_16bit - min_16bit))
    return image_float


def DepthNorm(depth, minDepth=0.0, maxDepth=10.0, doNorm=False):
    if doNorm:
        depth[depth < minDepth] = minDepth
        depth[depth > maxDepth] = maxDepth
        return maxDepth / depth
    else:
        # depth[depth < minDepth] = minDepth
        # depth[depth > maxDepth] = maxDepth
        return depth


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


# def scale_up(scale, images):
#     from skimage.transform import resize
#     scaled = []
#
#     for i in range(len(images)):
#         img = images[i]
#         output_shape = (scale * img.shape[0], scale * img.shape[1])
#         scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))
#
#     return np.stack(scaled)


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def compute_errors(gt, pred):

    # To avoid negative pred to ruin the testing process
    pred[pred == 0] = 1e-6
    gt[gt == 0] = 1e-6

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # rmse_log = (np.log(gt) - np.log(pred)) ** 2
    # rmse_log = np.sqrt(rmse_log.mean())

    # err = np.log(pred) - np.log(gt)
    # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return a1, a2, a3, abs_rel, rmse, log_10

# def compute_errors_mask(gt, pred):
def compute_errors_mask(gt, pred):
    mask_gt = np.where(gt > 0, gt, np.nan)
    mask_pred = np.where(pred > 0, pred, np.nan)

    # # To avoid negative pred to ruin the testing process
    # mask_pred = np.where(mask_pred == 0, 1e-6, mask_pred)
    # mask_gt = np.where(mask_gt == 0, 1e-6, mask_gt)

    # np.nanmean --> ignore nan
    # mask all negative values
    thresh = np.maximum((mask_gt / mask_pred), (mask_pred / mask_gt))
    a1 = np.nanmean(thresh < 1.25)
    a2 = np.nanmean(thresh < 1.25 ** 2)
    a3 = np.nanmean(thresh < 1.25 ** 3)

    import evaluate_depth_insight
    evaluate_depth_insight.count_img += 1
    '''
    # draw histogram # 
    import evaluate_depth_insight
    evaluate_depth_insight.count_img += 1
    threshold_values = thresh[thresh > 1.25]
    evaluate_depth_insight.error_list.extend(threshold_values)
    '''


    abs_rel = np.nanmean(np.abs(mask_gt - mask_pred) / mask_gt)
    # sq_rel = np.nanmean(((mask_gt - mask_pred) ** 2) / mask_gt)

    rmse = (mask_gt - mask_pred) ** 2
    rmse = np.sqrt(np.nanmean(rmse))

    log_10 = np.nanmean(np.abs(np.log10(mask_gt) - np.log10(mask_pred)))

    return a1, a2, a3, abs_rel, rmse, log_10

# def compute_errors(gt, pred):
#
#     '''
#     remove mask_top mask because it is hard to control due to png or exr inputs
#     pre-proess the data before training instead
#     '''
#
#     # mask outliers
#     mask_bottom = 0.5 > gt
#     # mask_bottom = 0 > gt
#     mask_top = gt > 10
#     # mask_top = gt > 20
#     # mask_top = gt > 9999
#     gt = np.ma.array(np.ma.array(gt, mask=mask_bottom), mask=mask_top)
#     pred = np.ma.array(pred, mask=gt.mask)
#
#
#     thresh = np.maximum((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()
#     abs_rel = np.mean(np.abs(gt - pred) / gt)
#     rmse = (gt - pred) ** 2
#     rmse = np.sqrt(rmse.mean())
#     # need to avoid 'nan' value (calculation between 'inf'œ)
#     log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
#     return a1, a2, a3, abs_rel, rmse, log_10

def save_output(batch_idx, img, depth, pre_depth, path=''):
    for i in range(len(img)):
        cv2.imwrite((path + '/rgb_' + str(batch_idx) + '_' + str(i)) + '.png', img[i].transpose(1, 2, 0))
        imageio.imwrite(path + '/'+ 'depth_' + str(batch_idx) + '_' + str(i) + '.exr', depth[i].transpose(1, 2, 0))
        imageio.imwrite(path + '/pre_depth_' + str(batch_idx) + '_' + str(i) + '.exr', pre_depth[i].transpose(1, 2, 0))

def show_output(img, depth, pre_depth):
    for i in range(len(img)):
        plt.imshow(img[i].transpose(1, 2, 0))
        plt.show()
        plt.imshow(depth[i].transpose(1, 2, 0))
        plt.show()
        plt.imshow(pre_depth[i].transpose(1, 2, 0))
        plt.show()

def evaluate_batch(model, test_loader, batch_size, minDepth, maxDepth, crop=None, verbose=False, is_da=False):
    # Load data
    # test_loader = getTrainingTestingData(batch_size=batch_size, csv_file=test_data_csv)

    start = time.time()
    print('Testing...')

    # Start Testing...
    result_list = []

    # N = len(test_loader)

    # Switch to evaluate mode
    model.eval()

    a1_list = []
    a2_list = []
    a3_list = []
    rel_list = []
    rms_list = []
    log10_list = []


    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):

            # Prepare sample and target
            if str(device) == 'cpu':
                image = torch.autograd.Variable(
                    sample_batched['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
                depth = torch.autograd.Variable(
                    sample_batched['depth'].to(device))  # ndarray: torch.Size([4, 256, 512])
            else:
                image = torch.autograd.Variable(sample_batched['image'].cuda())
                depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            if depth.shape[1] != 1:
                depth = depth.unsqueeze(dim=1)
                # depth = depth.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

            # Normalize depth
            depth_n = DepthNorm(depth, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)

            # Predict
            # output = model(image)
            # for ndarray type image
            if is_da:
                output, _ = model(image.float())
            else:
                output = model(image.float())
            # output_original, output = model(image.float())
            if isinstance(output, list):
                output_original, output = output

            # detach output
            output = output.detach().cpu().numpy()
            true_y = depth_n.detach().cpu().numpy()


            '''
            Plot Images for Saturation
            
            root_path = ''
            np.save(root_path + 'saturation_random_matrix.npy', RandomPhase)
            
            feature_type = 'saturation'
            
            plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
            plt.imsave(root_path + feature_type + '_sample.png', image.squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8), dpi=300)
            plt.show()
            
            plt.imshow(true_y.squeeze())
            plt.imsave(root_path + feature_type + '_gt.png', true_y.squeeze(), dpi=300)
            plt.show()
            
            plt.imshow(output.squeeze())
            plt.imsave(root_path + feature_type + '_output.png', output.squeeze(), dpi=300)
            plt.show()
            '''



            '''
            Print Images
            
            root_path = ''
            
            import evaluate_depth_insight
            img_type = 'shape2'
            plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
            plt.imsave(root_path + str(img_type) + '_'+ str(evaluate_depth_insight.count_img) +  '_sample.png', image.squeeze().permute(1, 2, 0).detach().cpu().numpy(), dpi=300)
            plt.show()
            plt.imshow(true_y.squeeze())
            plt.imsave(root_path + str(img_type) + '_'+ str(evaluate_depth_insight.count_img) +  '_gt.png', true_y.squeeze(), dpi=300)
            plt.show()
            plt.imshow(output.squeeze())
            plt.imsave(root_path + str(img_type) + '_' + str(evaluate_depth_insight.count_img) +  '_output.png', output.squeeze(), dpi=300)
            plt.show()
            
            # for Texture
            plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
            plt.imsave(root_path + 'texture_sample.png', image.squeeze().permute(1, 2, 0).detach().cpu().numpy(), dpi=300)
            plt.show()
            
            plt.imshow(true_y.squeeze(), cmap='viridis')
            plt.imsave(root_path + 'texture_gt.png', true_y.squeeze(), dpi=300, cmap='viridis')
            plt.show()
            
            plt.imshow(output.squeeze(), cmap='viridis')
            plt.imsave(root_path + 'texture_output.png', output.squeeze(), dpi=300, cmap='viridis')
            plt.show()
            
            # for Texture
            img = cv2.imread(sample_batched['img_path'][0].split(' ')[0], cv2.IMREAD_UNCHANGED)
            plt.imshow(img, cmap='gray')
            plt.imsave(root_path + 'texture_img.png', img, dpi=300, cmap='gray')
            plt.show()
            
            import json
            with open(sample_batched['img_path'][0].split(' ')[-1].replace('\n', '').replace('.png', '.json'), 'r') as json_file:
                recover_list_shuffled = json.load(json_file)
            with open('texture_matrix.json', 'w') as json_file:
                json.dump(recover_list_shuffled, json_file)
            
            # for Phase Scrambling
            RandomPhase = np.load(sample_batched['img_path'][0].split(' ')[-1].replace('\n', '').replace('.png', '.npy'))
            np.save(root_path + 'color_matrix.npy', RandomPhase)
            
            plt.imshow(image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
            plt.imsave(root_path + 'colour_sample.png', image.squeeze().permute(1, 2, 0).detach().cpu().numpy(), dpi=300)
            plt.show()
            
            plt.imshow(true_y.squeeze())
            plt.imsave(root_path + 'colour_gt.png', true_y.squeeze(), dpi=300)
            plt.show()
            
            plt.imshow(output.squeeze())
            plt.imsave(root_path + 'colour_output.png', output.squeeze(), dpi=300)
            plt.show()
            '''

            # Crop based on Eigen et al. crop
            if crop is not None:
                true_y = true_y[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
                output = output[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            import evaluate_depth_insight
            if evaluate_depth_insight.count_img == 100:
                print('')
            # Compute errors per image in batch
            for j in range(len(true_y)):
                gtDepth = true_y[j]
                prediction = output[j]

                # Resize to ground truth shape
                gtDepth = resize(gtDepth, prediction.shape, preserve_range=False, mode='reflect', anti_aliasing=True)

                # Compute errors for this image
                # errors = compute_errors(gtDepth, prediction)
                errors = compute_errors_mask(gtDepth, prediction)
                # errors = compute_errors_mask(gtDepth, prediction)

                a1_list.append(errors[0])
                a2_list.append(errors[1])
                a3_list.append(errors[2])
                rel_list.append(errors[3])
                rms_list.append(errors[4])
                log10_list.append(errors[5])

    e = [np.mean(a1_list), np.mean(a2_list), np.mean(a3_list), np.mean(rel_list), np.mean(rms_list), np.mean(log10_list)]
    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(e[0], e[1], e[2], e[3], e[4], e[5]))

    end = time.time()
    print('\nTest time', end - start, 's')

    return e

def evaluate_batch_single(model, test_loader, batch_size, minDepth, maxDepth, crop=None, verbose=False, is_da=False):
    # Load data
    # test_loader = getTrainingTestingData(batch_size=batch_size, csv_file=test_data_csv)

    start = time.time()
    print('Testing...')

    # Start Testing...
    result_list = []

    # N = len(test_loader)

    # Switch to evaluate mode
    model.eval()

    a1_list = []
    a2_list = []
    a3_list = []
    rel_list = []
    rms_list = []
    log10_list = []

    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):

            # Prepare sample and target
            if str(device) == 'cpu':
                image = torch.autograd.Variable(
                    sample_batched['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
                depth = torch.autograd.Variable(
                    sample_batched['depth'].to(device))  # ndarray: torch.Size([4, 256, 512])
            else:
                image = torch.autograd.Variable(sample_batched['image'].cuda())
                depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            if depth.shape[1] != 1:
                depth = depth.unsqueeze(dim=1)
                # depth = depth.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

            # Normalize depth
            depth_n = DepthNorm(depth, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)

            # Predict
            # output = model(image)
            # for ndarray type image
            if is_da:
                output, _ = model(image.float())
            else:
                output = model(image.float())
            # output_original, output = model(image.float())
            if isinstance(output, list):
                output_original, output = output

            # detach output
            output = output.detach().cpu().numpy()
            # true_y = depth_n.detach().cpu().numpy() / 255.0 # from uint8 to float32
            true_y = depth_n.detach().cpu().numpy()

            # Crop based on Eigen et al. crop
            if crop is not None:
                true_y = true_y[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
                output = output[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

            # Compute errors per image in batch
            for j in range(len(true_y)):
                gtDepth = true_y[j]
                prediction = output[j]

                # Resize to ground truth shape
                gtDepth = resize(gtDepth, prediction.shape, preserve_range=False, mode='reflect', anti_aliasing=True)

                # Compute errors for this image
                # errors = compute_errors(gtDepth, prediction)
                errors = compute_errors_mask(gtDepth, prediction)

                a1_list.append(errors[0])
                a2_list.append(errors[1])
                a3_list.append(errors[2])
                rel_list.append(errors[3])
                rms_list.append(errors[4])
                log10_list.append(errors[5])

    e = [np.mean(a1_list), np.mean(a2_list), np.mean(a3_list), np.mean(rel_list), np.mean(rms_list), np.mean(log10_list)]
    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(e[0], e[1], e[2], e[3], e[4], e[5]))

    end = time.time()
    print('\nTest time', end - start, 's')

    return e


def evaluate_batch_combined(model, test_loader_colour, test_loader_saturation, test_loader_texture, test_loader_shape,
                   batch_size, minDepth, maxDepth, crop=None, verbose=False, is_da=False):
    # Load data
    # test_loader = getTrainingTestingData(batch_size=batch_size, csv_file=test_data_csv)

    start = time.time()
    print('Testing...')

    # Start Testing...
    result_list = []

    # N = len(test_loader)

    # Switch to evaluate mode
    model.eval()

    a1_list = []
    a2_list = []
    a3_list = []
    rel_list = []
    rms_list = []
    log10_list = []

    with torch.no_grad():
        for i, (
        sample_batched_colour, sample_batched_saturation, sample_batched_texture, sample_batched_shape) in enumerate(
                zip(test_loader_colour, test_loader_saturation, test_loader_texture, test_loader_shape)):

            # Prepare sample and target
            if str(device) == 'cpu':
                image_colour = torch.autograd.Variable(sample_batched_colour['image'].to(device))
                depth_colour = torch.autograd.Variable(sample_batched_colour['depth'].to(device))
            else:
                image_colour = torch.autograd.Variable(sample_batched_colour['image'].cuda())
                depth_colour = torch.autograd.Variable(sample_batched_colour['depth'].cuda(non_blocking=True))

            if str(device) == 'cpu':
                image_saturation = torch.autograd.Variable(sample_batched_saturation['image'].to(device))
                depth_saturation = torch.autograd.Variable(sample_batched_saturation['depth'].to(device))
            else:
                image_saturation = torch.autograd.Variable(sample_batched_saturation['image'].cuda())
                depth_saturation = torch.autograd.Variable(sample_batched_saturation['depth'].cuda(non_blocking=True))

            if str(device) == 'cpu':
                image_texture = torch.autograd.Variable(sample_batched_texture['image'].to(device))
                depth_texture = torch.autograd.Variable(sample_batched_texture['depth'].to(device))
            else:
                image_texture = torch.autograd.Variable(sample_batched_texture['image'].cuda())
                depth_texture = torch.autograd.Variable(sample_batched_texture['depth'].cuda(non_blocking=True))

            if str(device) == 'cpu':
                image_shape = torch.autograd.Variable(sample_batched_shape['image'].to(device))
                depth_shape = torch.autograd.Variable(sample_batched_shape['depth'].to(device))
            else:
                image_shape = torch.autograd.Variable(sample_batched_shape['image'].cuda())
                depth_shape = torch.autograd.Variable(sample_batched_shape['depth'].cuda(non_blocking=True))

            if depth_colour.shape[1] != 1:
                depth_colour = depth_colour.unsqueeze(dim=1)
                depth_saturation = depth_saturation.unsqueeze(dim=1)
                depth_texture = depth_texture.unsqueeze(dim=1)
                depth_shape = depth_shape.unsqueeze(dim=1)

                # depth = depth.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

            # Normalize depth
            depth_n_colour = DepthNorm(depth_colour, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)
            depth_n_saturation = DepthNorm(depth_saturation, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)
            depth_n_texture = DepthNorm(depth_texture, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)
            depth_n_shape = DepthNorm(depth_shape, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)

            # Predict
            output = model(image_colour.float(), image_saturation.float(), image_texture.float(), image_shape.float())

            # output_original, output = model(image.float())
            if isinstance(output, list):
                output_original, output = output

            depth_n = torch.cat((depth_n_colour, depth_n_saturation, depth_n_texture, depth_n_shape), dim=1)

            # detach output
            output = output.detach().cpu().numpy()
            true_y = depth_n.detach().cpu().numpy()

            # Crop based on Eigen et al. crop
            if crop is not None:
                true_y = true_y[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
                output = output[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

            # Compute errors per image in batch
            for j in range(len(true_y)):
                gtDepth = true_y[j]
                prediction = output[j]

                # Resize to ground truth shape
                gtDepth = resize(gtDepth, prediction.shape, preserve_range=False, mode='reflect', anti_aliasing=True)

                # Compute errors for this image
                errors = compute_errors(gtDepth, prediction)

                a1_list.append(errors[0])
                a2_list.append(errors[1])
                a3_list.append(errors[2])
                rel_list.append(errors[3])
                rms_list.append(errors[4])
                log10_list.append(errors[5])

    e = [np.mean(a1_list), np.mean(a2_list), np.mean(a3_list), np.mean(rel_list), np.mean(rms_list),
         np.mean(log10_list)]
    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(e[0], e[1], e[2], e[3], e[4], e[5]))

    end = time.time()
    print('\nTest time', end - start, 's')

    return e


def evaluate(model, test_loader, batch_size, minDepth, maxDepth, crop=None, verbose=False, is_da=False):
    # Load data
    # test_loader = getTrainingTestingData(batch_size=batch_size, csv_file=test_data_csv)

    start = time.time()
    print('Testing...')

    # Start Testing...
    result_list = []

    # N = len(test_loader)

    # Switch to evaluate mode
    model.eval()

    predictions = []
    testSetDepths = []

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        if str(device) == 'cpu':
            image = torch.autograd.Variable(
                sample_batched['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
            depth = torch.autograd.Variable(
                sample_batched['depth'].to(device))  # ndarray: torch.Size([4, 256, 512])
        else:
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        if depth.shape[1] != 1:
            depth = depth.unsqueeze(dim=1)
            # depth = depth.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

        # Normalize depth
        depth_n = DepthNorm(depth, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)

        # Predict
        # output = model(image)
        # for ndarray type image
        with torch.no_grad():
            if is_da:
                output, _ = model(image.float())
            else:
                output = model(image.float())
            # output_original, output = model(image.float())
            if isinstance(output, list):
                output_original, output = output

        # detach output
        output = output.detach().cpu().numpy()
        true_y = depth_n.detach().cpu().numpy()

        # Crop based on Eigen et al. crop
        if crop is not None:
            true_y = true_y[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            output = output[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append(output[j])
            testSetDepths.append(true_y[j])

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)


    # ndarray resize
    from skimage.transform import resize
    gtDepthSet = resize(testSetDepths, predictions.shape, preserve_range=False, mode='reflect', anti_aliasing=True)

    e = compute_errors(gtDepthSet, predictions)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4],
                                                                                  e[5]))

    end = time.time()
    print('\nTest time', end - start, 's')

    return e


def evaluate_show(model, test_loader, batch_size, minDepth, maxDepth, crop=None, verbose=False, is_da=False):
    # Load data
    # test_loader = getTrainingTestingData(batch_size=batch_size, csv_file=test_data_csv)

    start = time.time()
    print('Showing ...')

    # Start Testing...
    result_list = []

    # N = len(test_loader)

    # Switch to evaluate mode
    model.eval()

    predictions = []
    testSetDepths = []

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        if str(device) == 'cpu':
            image = torch.autograd.Variable(
                sample_batched['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
            depth = torch.autograd.Variable(
                sample_batched['depth'].to(device))  # ndarray: torch.Size([4, 256, 512])
        else:
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        if depth.shape[1] != 1:
            depth = depth.unsqueeze(dim=1)
            # depth = depth.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

        # Normalize depth
        depth_n = DepthNorm(depth, minDepth=minDepth, maxDepth=maxDepth, doNorm=False)

        # Predict
        # output = model(image)
        # for ndarray type image
        with torch.no_grad():
            if is_da:
                output, _ = model(image.float())
            else:
                output = model(image.float())

            showed_image = (output.squeeze().detach().cpu())
            plt.imshow(showed_image)
            plt.show()



    # ndarray resize
    from skimage.transform import resize
    gtDepthSet = resize(testSetDepths, predictions.shape, preserve_range=False, mode='reflect', anti_aliasing=True)

    e = compute_errors(gtDepthSet, predictions)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4],
                                                                                  e[5]))

    end = time.time()
    print('\nTest time', end - start, 's')

    return e


def img_normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    img = img.astype("float") / 255.0
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    return img



