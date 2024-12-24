import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import json

from micclip.modeling import registry

from .lib.utils.tools import *
from .lib.utils.learning import *
from .lib.model.loss import *
from .lib.data.dataset_action import NTURGBD
from .lib.model.model_action import ActionNet


@registry.MOTION.register("Motion")
class load_3d_motion(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        pretrained = './data/models/pretrained_models/MB_train_NTU60_xsub/best_epoch.bin'
        selection = ''
        chk_filename = pretrained
        print('Loading backbone', chk_filename)
        model_backbone = load_backbone()
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_backbone = load_pretrained_weights(model_backbone, checkpoint)
        self.model = ActionNet(backbone=model_backbone, dim_rep=512, num_classes=16, dropout_ratio=0.5, version='embedding', hidden_dim=2048, num_joints=17)
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.device = device
        self.head = nn.Linear(2048, 512)
        self.motion_to_image = nn.Linear(512, 296)

    def halpe2h36m(self, x):
        '''
        Input: x (T x V x C)  
        //Halpe 26 body keypoints
        {0,  "Nose"},
        {1,  "LEye"},
        {2,  "REye"},
        {3,  "LEar"},
        {4,  "REar"},
        {5,  "LShoulder"},
        {6,  "RShoulder"},
        {7,  "LElbow"},
        {8,  "RElbow"},
        {9,  "LWrist"},
        {10, "RWrist"},
        {11, "LHip"},
        {12, "RHip"},
        {13, "LKnee"},
        {14, "Rknee"},
        {15, "LAnkle"},
        {16, "RAnkle"},
        {17,  "Head"},
        {18,  "Neck"},
        {19,  "Hip"},
        {20, "LBigToe"},
        {21, "RBigToe"},
        {22, "LSmallToe"},
        {23, "RSmallToe"},
        {24, "LHeel"},
        {25, "RHeel"},
        '''
        T, V, C = x.shape
        y = np.zeros([T,17,C])
        y[:,0,:] = x[:,19,:]
        y[:,1,:] = x[:,12,:]
        y[:,2,:] = x[:,14,:]
        y[:,3,:] = x[:,16,:]
        y[:,4,:] = x[:,11,:]
        y[:,5,:] = x[:,13,:]
        y[:,6,:] = x[:,15,:]
        y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
        y[:,8,:] = x[:,18,:]
        y[:,9,:] = x[:,0,:]
        y[:,10,:] = x[:,17,:]
        y[:,11,:] = x[:,5,:]
        y[:,12,:] = x[:,7,:]
        y[:,13,:] = x[:,9,:]
        y[:,14,:] = x[:,6,:]
        y[:,15,:] = x[:,8,:]
        y[:,16,:] = x[:,10,:]
        return y

    def sortFn(self, dict):
        return dict['idx']

    def get_embedding(self, movie_name, img_id, person_bbox):
        
        motion = []
        
        for idx, movie in enumerate(movie_name):
            json_path = './data/keypoints/'
            file_path = './data/jhmdb/videos/' + movie_name[idx] + '/' 
            json_path = json_path + movie + '/alphapose-results.json'

            with open(json_path, "r") as read_file:
                results = json.load(read_file)

            results.sort(key=self.sortFn)
            
            action_dict = {}
            bbox_dict = {}

            for frames in results:
                if (frames["image_id"][:-4] == str(img_id[idx]).zfill(5)):
                    person_id = frames["idx"]
                    if (type(person_id) != type(5)):
                        for person_idx in person_id:
                            if person_idx not in action_dict.keys():
                                action_dict[person_idx] = []
                                bbox_dict[str(frames["box"][0])] = person_idx
                    elif (person_id not in action_dict.keys()): 
                        action_dict[person_id] = []
                        bbox_dict[str(frames["box"][0])] = person_id

                    # original
                    for other_frames in results:

                        # if idx is list
                        if (type(other_frames["idx"]) != type(5)):
                            for other_idx in other_frames["idx"]:
                                if (type(person_id) != type(5)):
                                    for person_idx in person_id:
                                        if (other_idx == person_idx):
                                            action_dict[person_idx].append(np.array(other_frames["keypoints"]).reshape(-1,3))
                                else:
                                    if (other_idx == person_id):
                                        action_dict[person_id].append(np.array(other_frames["keypoints"]).reshape(-1,3))

                        # if idx  is int
                        else: 
                            if (type(person_id) != type(5)):
                                for person_idx in person_id:
                                    if (other_frames["idx"] == person_idx):
                                        action_dict[person_idx].append(np.array(other_frames["keypoints"]).reshape(-1,3))
                            else:
                                if (other_frames["idx"] == person_id):
                                    action_dict[person_id].append(np.array(other_frames["keypoints"]).reshape(-1,3))
                    

                        
            for person in person_bbox[idx]:

                closet = [abs(person[0] - float(keys)) for keys in bbox_dict.keys()]
                if (len(closet) == 0): kpts = np.zeros((1,17,3))
                else:
                    box_keys = [k for k in bbox_dict.keys()]
                    tid = box_keys[closet.index(min(closet))]

                    
                    kpts = action_dict[bbox_dict[tid]]
                    kpts = np.array(kpts)
                    kpts = self.halpe2h36m(kpts)

                output = self.model(torch.tensor(np.array(kpts)).to(self.device))
                output = self.head(output).squeeze(0)
                motion.append(output)


        motion = torch.stack(motion).to(self.device)
        return motion

def motion_3d(cfg, device):
    func = registry.MOTION[cfg.MODEL.ROI_ACTION_HEAD.MOTION]
    return func(cfg, device)






