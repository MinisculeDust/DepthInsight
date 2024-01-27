from phaseScramble_depth import phaseScramble_depth, phaseScramble_depth_replacePath
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filenames_file',
                        default='nyu_large_test_10p.txt', type=str,
                        help='NYU images')
parser.add_argument('--originalKeyword', default='/NYU_large/', type=str,)
parser.add_argument('--savingKeyword', default='/NYU_large_RGB_Scrambled/', type=str,)
args = parser.parse_args()

with open(args.filenames_file, 'r') as f:
    filenames = f.readlines()

for i, item in enumerate(filenames):
    phaseScramble_depth_replacePath(filenames[i].split(' ')[0], filenames[i].split(' ')[1], args.originalKeyword, args.savingKeyword)


print('Finished ' + str(len(filenames)) + ' scenes')