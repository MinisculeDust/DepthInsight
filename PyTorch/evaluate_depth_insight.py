# -*- coding: utf-8 -*-
import argparse
import time
import wandb
import torch
import torch.nn as nn
from utils import AverageMeter, SILogLoss
import evaluate_gpu
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


PROJECT = "Depth_Insight"

count_img = 0
error_list = []

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument("--trained_model",
                        default="",
                        type=str, help='path of trained model')
    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='max learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--validate_every', default=5, type=int, help='validation period')
    parser.add_argument("--name", default="DepthInsight", type=str, help='experiment name on WandB')
    parser.add_argument("--root", default=".", type=str, help="Root folder to save data in")
    parser.add_argument("--num_threads", "--workers", default=4, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='NYU', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='', type=str, help="path to dataset")
    parser.add_argument("--gt_path", default='', type=str, help="path to dataset")
    parser.add_argument('--featureType', default='pseudo_rgb', type=str, help='Types of inputs')
    parser.add_argument('--patchSize', default=16, type=int, help='patch size of texture feature')
    parser.add_argument('--filenames_file_eval', default='',
                        type=str, help='large NYU Direct Saturation Testing dataset (10p) images')
    parser.add_argument('--scheduler', type=bool, help='use scheduler or not', default=False)
    parser.add_argument('--scheduler_patience', type=int, help='scheduler patience', default=10)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--logging', type=bool, default=False, help='log with WandB')
    parser.add_argument('--appendix', type=str, default='', help='notes for the experiment')
    args = parser.parse_args()

    from dataloader import DepthDataLoader
    featureType_single = []
    featureType_multi = ['rgb', 'pseudo_rgb', 'colour', 'texture', 'shape', 'colour_grayscale', 'saturation',
                         'single_grayscale', 'shape2']
    from model_ResNet50 import Model

    print('args.filenames_file_eval', args.filenames_file_eval)

    '''
    The model is automatically prefixed with "module.", 
    but the loaded model parameters are not prefixed with "module.", 
    so the parameter names need to be changed
    '''
    model = Model()
    # load parameters
    loaded_state_dict = torch.load(args.trained_model)
    # change parameter names
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        new_key = key.replace("module.", "")  # remove "module."
        new_state_dict[new_key] = value
    # load edited params
    model.load_state_dict(new_state_dict)


    # checking muti-GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = args.bs
    prefix = 'ResNet_' + str(batch_size)

    # Loss
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    silog_criterion = SILogLoss()

    args.epoch = 0

    test_loader = DepthDataLoader(args, 'online_eval', args.featureType).data

    # evaluating Process
    with torch.autograd.set_detect_anomaly(False):
        # global information
        iters = len(test_loader)
        step = args.epoch * iters

        # Start evaluating ...
        for epoch in range(args.epochs):
            if args.logging:
                wandb.log({"Epoch": epoch}, step=step)

            batch_time = AverageMeter()
            losses = AverageMeter()
            N = len(test_loader)

            # Switch to eval mode
            model.eval()

            end = time.time()

            # 使用tqdm包裹test_loader
            if args.featureType in featureType_single:
                e = evaluate_gpu.evaluate_batch_single(model, test_loader, minDepth=args.min_depth,
                                                           maxDepth=args.max_depth,
                                                           crop=None, batch_size=args.bs)
            else:
                e = evaluate_gpu.evaluate_batch(model, test_loader, minDepth=args.min_depth, maxDepth=args.max_depth,
                                                    crop=None,
                                                    batch_size=args.bs)
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms',
                                                                              'log_10'))
            print(
                    "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))









