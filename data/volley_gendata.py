'''
volley_gendata.py is adopted from https://github.com/hongluzhou/composer/blob/main/datasets/volleyball.py
'''

import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap
import glob
from collections import defaultdict
from einops import rearrange, repeat
import cv2

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

normal_splits = {
    'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39,
                40, 41, 42, 48, 50, 52, 53, 54, 0, 2, 8, 12, 17, 19, 24, 26,
                27, 28, 30, 33, 46, 49, 51],
    'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
}

olympic_splits = {
    'train': [1, 2, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'test': [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21, 22, 23, 24, 25, 26, 27]
}

idx2class = {
            0: {'r_set', 'r-set'},
            1: {'l_set', 'l-set'},
            2: {'r_spike', 'r-spike'},
            3: {'l_spike', 'l-spike'},
            4: {'r_pass', 'r-pass'},
            5: {'l_pass', 'l-pass'},
            6: {'r_winpoint', 'r-winpoint'},
            7: {'l_winpoint', 'l-winpoint'}
        }

class2idx = dict()
for k in idx2class:
    for v in idx2class[k]:
        class2idx[v] = k


classidx_horizontal_flip_augment = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    6: 7,
    7: 6
}

bone_r = {
    2:0,
    4:2,
    5:6,
    8:6,
    10:8,
    12:6,
    14:12,
    16:14
}

bone_l = {
    1:0,
    3:1,
    5:7,
    9:7,
    11:5,
    13:11,
    15:13
}


def disentangle_gl(raw_data_seq): 
    C, T, J, P = raw_data_seq.shape
    
    # remove face keypoints
    raw_data_seq = np.concatenate((raw_data_seq[:,:,0,:].reshape([C,T,1,P]), raw_data_seq[:,:,5:,:]), axis=2)
    l_s_joint = 1 #5
    r_s_joint = 2 #6
    
    l_s = raw_data_seq[:,:,l_s_joint,:].reshape([C,T,1,P])
    r_s = raw_data_seq[:,:,r_s_joint,:].reshape([C,T,1,P])
    center = (l_s + r_s)/2.0
    dist_2_center = raw_data_seq - center

    # add global translation to the first joint
    initial_center = center[:,0,:,:].reshape([C,1,1,P])
    center_translation = center - initial_center
    data = np.concatenate((center_translation, dist_2_center), axis=2)

    return data


def remove_avg_movement(gl_data_seq):
    C, T, J, P = gl_data_seq.shape
    #first_frame_center = gl_data_seq[:,0,0,:]

    p_centers = gl_data_seq[:,:,0,:]
    avg_movement = np.average(p_centers, axis=2)
    avg_movement = avg_movement.reshape([C,T,1])
    avg_movement = np.repeat(avg_movement, repeats=P, axis=2)
    gl_data_seq[:,:,0,:] = gl_data_seq[:,:,0,:] - avg_movement

    return gl_data_seq


def gendata(data_path,
            out_path,
            benchmark='normal',
            split='test'):

    if benchmark=='normal':
        dataset_splits = normal_splits
    else:
        dataset_splits = olympic_splits

    # clip path
    clip_paths = []
    for idx in dataset_splits[split]:
        clip_paths.extend(glob.glob(os.path.join(data_path, 'joints', str(idx), '*.pickle')))


    labels = []
    data = []
    # clip label
    clip_annotations= defaultdict()
    for annot_file in glob.glob(os.path.join(data_path, 'videos/*/annotations.txt')):
        video = annot_file.split('/')[-2]
        with open(annot_file, 'r') as f:
            lines = f.readlines()
        for l in lines:
            clip, label = l.split()[0].split('.jpg')[0], l.split()[1]
            clip_annotations[(video, clip)] = class2idx[label]  

    for count, path in enumerate(clip_paths):
        video, clip = path.split('/')[-2], path.split('/')[-1].split('.pickle')[0]
        print(path, video, clip)


        # joints seq
        raw_data = pickle.load(open(path, "rb"))
        frames = sorted(raw_data.keys())

        raw_data_seq = []
        for t in frames:
            raw_data_seq.append(raw_data[t][:,:,:2])
        raw_data_seq = np.array(raw_data_seq)
        raw_data_seq = rearrange(raw_data_seq, 't p j c -> c t j p') # 2D frames joints people

        data_seq = disentangle_gl(raw_data_seq)
        data_seq = remove_avg_movement(data_seq)
        data.append(data_seq)
        # label
        labels.append(clip_annotations[(video, clip)])

        if split == 'train':
            flip_seq = data_seq.copy()
            flip_seq[0,:,:,:] = -flip_seq[0,:,:,:]
            data.append(flip_seq)
            # label
            labels.append(classidx_horizontal_flip_augment[clip_annotations[(video, clip)]])
            

    with open('{}/{}_label.pkl'.format(out_path, split), 'wb') as f:
        pickle.dump((labels), f)

    with open('{}/{}_data.pkl'.format(out_path, split), 'wb') as f:
        pickle.dump((data), f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Volley Ball Data Converter.')
    parser.add_argument('--data_path', default='./volleyball')
    parser.add_argument('--out_folder', default='./preprocessed')

    benchmark = ['normal']
    split = ['train', 'test']

    arg = parser.parse_args()

    for b in benchmark:
        for p in split:
            out_path = arg.out_folder
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                benchmark=b,
                split=p)
    