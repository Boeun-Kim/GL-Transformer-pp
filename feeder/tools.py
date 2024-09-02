'''
This file is for data augmentation
'''

import numpy as np
import random
import cv2


def shear(data_numpy, r=0.5):

    center_trans = data_numpy[:,:, 0, :].copy()
    data_numpy[:,:,0,:] = 0.0

    s1 = random.uniform(-r, r)
    s2 = random.uniform(-r, r)

    R = np.array([[1,   s1],
                  [s2,  1 ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)

    data_numpy[:,:,0,:] = center_trans
    
    return data_numpy


def interpolate(data_numpy, interpolate_ratio=0.0, max_frame=25):
    C, T, V, M = data_numpy.shape

    if np.random.rand() > 0.5:
        sign = -1
    else:
        sign = 1
    rand_ratio = np.random.rand()
    interpolate_size = int(T * (1 + sign*rand_ratio*interpolate_ratio))
    interpolate_size = max(1, interpolate_size)
    new_data = np.zeros((C, interpolate_size, V, M))

    for i in range(M):
        tmp = cv2.resize(data_numpy[:, :, :, i].transpose(
            [1, 2, 0]), (V, interpolate_size), interpolation=cv2.INTER_LINEAR)

        tmp = tmp.transpose([2, 0, 1])

        new_data[:, :, :, i] = tmp

    if new_data.shape[1] > max_frame:
        new_data = new_data[:, :max_frame, :, :]

    return new_data


def purturb(data_numpy, purturb_range=0.0):

    C, T, V, M = data_numpy.shape
    purturb = np.random.normal(0, purturb_range, (C,T,V,M))
    data_numpy = data_numpy + purturb

    return data_numpy

def flip_x(data_numpy):
    data_numpy[0,:,:,:] = -data_numpy[0,:,:,:]
    return data_numpy

def flip_y(data_numpy):
    data_numpy[1,:,:,:] = -data_numpy[1,:,:,:]
    return data_numpy

def change_order(data_numpy, order):
    data_numpy_copy = data_numpy.copy()
    data_numpy = data_numpy_copy[:,:,:,order]

    return data_numpy