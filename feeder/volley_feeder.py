import numpy as np
import pickle, torch
from . import tools
import glob
import os
from collections import defaultdict
from einops import rearrange, repeat



class FeederPretrain(torch.utils.data.Dataset):

    def __init__(self, data_path, shear_amplitude=0.3, interpolate_ratio=0.1, purturb_range=1.0, intervals=[1,5,10], split='train', img_w=1280, img_h=720, max_frame=25):
        self.data_path = data_path
        self.split = split
        self.img_w = img_w
        self.img_h = img_h
        self.max_frame = max_frame
        
        self.shear_amplitude = shear_amplitude
        self.interpolate_ratio = interpolate_ratio
        self.purturb_range = purturb_range
        self.intervals = intervals
        self.labels = []
        self.data = []
        
        self.load_data()


    def load_data(self):

        with open('{}/{}_label.pkl'.format(self.data_path, self.split), 'rb') as f:
            self.labels = pickle.load(f)

        with open('{}/{}_data.pkl'.format(self.data_path, self.split), 'rb') as f:
            self.data = pickle.load(f)
        
        
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):

        real_data = self.data[index]
        
        real_data_aug = self._aug(real_data)

        c, length, num_joints, num_people = real_data_aug.shape 
        data = np.ones((c, self.max_frame, num_joints, num_people))*999.9
        data[:,:length, :, :] = real_data_aug

        # calculate gt displacement
        rolls = self.intervals
        dis_list = []
        for roll in rolls:
            displacement = np.zeros_like(data)
            data_roll = np.roll(data, roll, axis=1)
            data_roll[:,:roll, :, :] = np.expand_dims(data[:, 0,:,:], axis=1)

            displacement = data_roll - data
            dis_list.append(displacement)

        dis_concat = np.concatenate(dis_list, axis=2)

        # calculate gt displacement magnitude & decide class
        roll_mags = self.intervals
        mag_list = []
        for roll_mag in roll_mags:
            data_roll = np.roll(data, roll_mag, axis=1)
            data_roll[:,:roll_mag, :, :] = np.expand_dims(data[:, 0,:,:], axis=1)

            displacement = data_roll - data
            displacement = np.power(displacement[0], 2)+np.power(displacement[1], 2)
            mag = np.sqrt(displacement)

            mag_quant = mag//1.0 +1
            mag_quant[mag_quant > 14] = 14
            mag_quant[mag==0.0] = 0
            if np.sum(data[:,0,:,1]) == 0:
                mag_quant[:,:,1] = 15
            
            #print(mag_quant)
            mag_quant = mag_quant.reshape(self.max_frame, -1)
            mag_quant = mag_quant.astype(int)

            mag_list.append(mag_quant)


        mag_gt = np.concatenate(mag_list, axis=1)

        # reshape
        data = data.reshape(c, self.max_frame, -1)
        dis_concat = dis_concat.reshape(c, dis_concat.shape[1], -1)

        ## decide class of displacement direction
        xyz_direction = np.zeros((c, dis_concat.shape[1], dis_concat.shape[2]), dtype=int)
        xyz_direction[dis_concat == 0] = 1
        xyz_direction[dis_concat > 0] = 2

        dir_gt = xyz_direction[0] + xyz_direction[1]*3

        return data, dir_gt, mag_gt

    def _aug(self, data_numpy):

        if self.purturb_range > 0:
            data_numpy = tools.purturb(data_numpy, self.purturb_range)

        if self.interpolate_ratio > 0:
            data_numpy = tools.interpolate(data_numpy, self.interpolate_ratio, self.max_frame)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy



class Feeder_actionrecog(torch.utils.data.Dataset):

    def __init__(self, data_path, shear_amplitude=0.3, interpolate_ratio=0.1, purturb_range=1.0, split='test', img_w=1280, img_h=720, max_frame=25):
        self.data_path = data_path
        self.split = split
        self.img_w = img_w
        self.img_h = img_h
        self.max_frame = max_frame
        
        self.purturb_range = purturb_range
        self.shear_amplitude = shear_amplitude
        self.interpolate_ratio = interpolate_ratio
        self.labels = []
        self.data = []
        
        self.load_data()


    def load_data(self):

        with open('{}/{}_label.pkl'.format(self.data_path, self.split), 'rb') as f:
            self.labels = pickle.load(f)

        with open('{}/{}_data.pkl'.format(self.data_path, self.split), 'rb') as f:
            self.data = pickle.load(f)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        real_data = self.data[index]

        real_data_aug = self._aug(real_data)

        c, length, num_joints, num_people = real_data_aug.shape 
        data = np.ones((c, self.max_frame, num_joints, num_people))*999.9
        data[:,:length, :, :] = real_data_aug

        data = data.reshape(c, self.max_frame, -1)

        return data, label

    def _aug(self, data_numpy):

        if self.purturb_range > 0:
            data_numpy = tools.purturb(data_numpy, self.purturb_range)

        if self.interpolate_ratio > 0:
            data_numpy = tools.interpolate(data_numpy, self.interpolate_ratio, self.max_frame)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy
