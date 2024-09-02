'''
learn_PTmodel.py is for pretraining GL-Transformer model
'''

import os
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pickle
from tensorboardX import SummaryWriter
import argparse
from torchsummary import summary
from feeder.volley_feeder import FeederPretrain
from model.pretrain import Pretrain
from arguments import parse_args
from einops import rearrange, repeat

def init_seed(seed=123):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

init_seed()

def load_data(is_train, data_path, batch_size, num_workers, shear_amplitude=-1, interpolate_ratio=-1, purturb_range = -1, intervals=[1,5,10], img_w=1280, img_h=720, max_frame=25):
    train_feeder_args = {'data_path': data_path,
                        'shear_amplitude': shear_amplitude,
                        'interpolate_ratio': interpolate_ratio,
                        'purturb_range' : purturb_range,
                        'intervals' : intervals,
                        'split' : 'train',
                        'img_w' : img_w,
                        'img_h' : img_h,
                        'max_frame': max_frame
                        }
    test_feeder_args = {'data_path': data_path,
                        'shear_amplitude': shear_amplitude,
                        'interpolate_ratio': interpolate_ratio,
                        'purturb_range' : purturb_range,
                        'intervals' : intervals,
                        'split' : 'test',
                        'img_w' : img_w,
                        'img_h' : img_h,
                        'max_frame': max_frame
                        }

    if is_train:
        data_loader = torch.utils.data.DataLoader(
            dataset=FeederPretrain(**train_feeder_args),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=init_seed
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=FeederPretrain(**test_feeder_args),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=init_seed
        )
        
    return data_loader


train_writer = SummaryWriter(os.path.join('output', 'train'), 'train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    f=open(args.save_path+'_learnPT.txt', 'w')

    print("pretrain max epoch: ", args.max_epoch)
    loader = load_data(is_train=True, data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, 
                        shear_amplitude=args.shear_amplitude, interpolate_ratio=args.interpolate_ratio, 
                        purturb_range = args.purturb_range, intervals=args.intervals, img_w=args.img_width, img_h=args.img_height, max_frame=args.num_frame)

    model = Pretrain
    pretrain_model = model(num_frame=args.num_frame, num_joint=args.num_joint, num_p=args.num_people, input_channel=args.input_channel, dim_emb=args.dim_emb, 
                            depth=args.depth, num_heads=args.num_heads, qkv_bias=args.qkv_bias, ff_expand=args.ff_expand, 
                            do_rate=args.do_rate ,attn_do_rate=args.attn_do_rate,
                            drop_path_rate=args.drop_path_rate, add_positional_emb=args.add_positional_emb, intervals=args.intervals, cross_rolls=args.cross_rolls)

    print("gpus: ", np.arange(args.gpus))
    pretrain_model = pretrain_model.to(device)
    if torch.cuda.device_count() > 1:
        pretrain_model = nn.DataParallel(pretrain_model, device_ids=np.arange(args.gpus).tolist())

    pretrain_model.train()
    #print(sum(p.numel() for p in pretrain_model.transformer.parameters() if p.requires_grad))

    #summary(pretrain_model, (2,24,168), batch_size=4)

    lr = args.lr
    optimizer = optim.AdamW(pretrain_model.parameters(), lr=args.lr, weight_decay=0.1)

    criterion = nn.CrossEntropyLoss()
    criterion_binary = nn.CrossEntropyLoss()
    criterion_MSE = nn.MSELoss()

    step = 0

    for epoch_idx in range(1, args.max_epoch + 1):

        
        pretrain_model.train()
        for position, dir_gt, mag_gt in loader:
            
            position = position.float().to(device)
            dir_gt= dir_gt.to(device)
            mag_gt= mag_gt.to(device)
            
            optimizer.zero_grad()
            
            out, out2 = pretrain_model(position)     

            x_m = position[:,0,:,0] != 999.9

            out = rearrange(out, 'b c f (j p) -> b c f j p', p=args.num_people)

            out_global = out[:,:,:,0,:].unsqueeze(3)  # global translation index: 0
            out_joints = out[:,:,:,1:,:]
            #out_joints = torch.cat((out[:,:,:,:20,:], out[:,:,:,21:,:]),axis=3)
            out_global = rearrange(out_global, 'b c f j p -> b f (j p) c',)
            out_joints = rearrange(out_joints, 'b c f j p -> b f (j p) c',)

            out_global = out_global[x_m]  # for calculating loss except [PAD] tokens
            out_global = rearrange(out_global, 'm j k -> (m j) k',)
            out_joints = out_joints[x_m]
            out_joints = rearrange(out_joints, 'm j k -> (m j) k',)

            dir_gt = rearrange(dir_gt, 'b f (j p) -> b f j p', p=args.num_people)
            dir_gt_global = dir_gt[:,:,0,:].unsqueeze(3)
            dir_gt_joint = dir_gt[:,:,1:,:]
            #dir_gt_joint = torch.cat((dir_gt[:,:,:20,:], dir_gt[:,:,21:,:]), axis=2)
            dir_gt_global = rearrange(dir_gt_global, 'b f j p -> b f (j p)')
            dir_gt_joint = rearrange(dir_gt_joint, 'b f j p -> b f (j p)')

            dir_gt_global = dir_gt_global[x_m]
            dir_gt_global = rearrange(dir_gt_global, 'm j -> (m j)',)
            dir_gt_joint = dir_gt_joint[x_m]
            dir_gt_joint = rearrange(dir_gt_joint, 'm j -> (m j)',)
            
            loss_dir_g = criterion(out_global, dir_gt_global)
            loss_dir = criterion(out_joints, dir_gt_joint)


            out2 = rearrange(out2, 'b c f (j p) -> b c f j p', p=args.num_people)
            out2_global = out2[:,:,:,0,:].unsqueeze(3)
            out2_joints = out2[:,:,:,1:,:]
            #out2_joints = torch.cat((out2[:,:,:,:20,:], out2[:,:,:,21:,:]),axis=3)
            out2_global = rearrange(out2_global, 'b c f j p -> b f (j p) c',)
            out2_joints = rearrange(out2_joints, 'b c f j p -> b f (j p) c',)
            
            out2_global = out2_global[x_m]
            out2_global = rearrange(out2_global, 'm j k -> (m j) k',)
            out2_joints = out2_joints[x_m]
            out2_joints = rearrange(out2_joints, 'm j k -> (m j) k',)

            mag_gt = rearrange(mag_gt, 'b f (j p) -> b f j p', p=args.num_people)
            mag_gt_global = mag_gt[:,:,0,:].unsqueeze(3)
            mag_gt_joint = mag_gt[:,:,1:,:]
            #mag_gt_joint = torch.cat((mag_gt[:,:,:20,:], mag_gt[:,:,21:,:]), axis=2)
            mag_gt_global = rearrange(mag_gt_global, 'b f j p -> b f (j p)')
            mag_gt_joint = rearrange(mag_gt_joint, 'b f j p -> b f (j p)')

            mag_gt_global = mag_gt_global[x_m]
            mag_gt_global = rearrange(mag_gt_global, 'm j -> (m j)',)
            mag_gt_joint = mag_gt_joint[x_m]
            mag_gt_joint = rearrange(mag_gt_joint, 'm j -> (m j)',)

            loss_mag_g = criterion(out2_global, mag_gt_global)
            loss_mag = criterion(out2_joints, mag_gt_joint)
            loss = (args.lambda_global*loss_dir_g+ loss_dir) + args.lambda_mag*(args.lambda_global*loss_mag_g + loss_mag)

            loss.backward()
            optimizer.step()

            train_writer.add_scalar('step_loss', loss, step)
            step += 1
            print(epoch_idx, '%f'%(loss), lr)
            f.write(str(epoch_idx) +'\t'+str(loss)+'\t'+str(lr)+'\n') 
            
        if epoch_idx % 30 == 0 or epoch_idx == 1:
 
            if torch.cuda.device_count() > 1:
                pretrain_model.module.save(epoch_idx, args.save_path)
            else:
                pretrain_model.save(epoch_idx, args.save_path)
        
            
        lr *= args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
    
    f.close()
