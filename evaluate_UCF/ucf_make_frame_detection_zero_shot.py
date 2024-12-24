#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import csv
import numpy as np
from tqdm import tqdm


# In[4]:


labels = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling','Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing','LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing','Skijet','SoccerJuggling','Surfing','TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
# our_test_labels = ['IceDancing', 'FloorGymnastics', 'SalsaSpin', 'SkateBoarding', 'SoccerJuggling', 'VolleyballSpiking']
# our_train_labels = ['Diving', 'CricketBowling', 'PoleVault', 'TennisSwing', 'BasketballDunk', 'Biking', 'Skijet', 'CliffDiving', 'LongJump', 'HorseRiding', 'Basketball', 'GolfSwing', 'Skiing', 'RopeClimbing', 'Surfing', 'Fencing', 'TrampolineJumping', 'WalkingWithDog']
our_test_labels = open("/work/sc19981018/webber/ucf_label_split/75vs25/2/test_label.txt", 'r').read().split("\n")
our_train_labels = open("/work/sc19981018/webber/ucf_label_split/75vs25/2/train_label.txt", 'r').read().split("\n")
objects = []

with (open("UCF101v2-GT.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile,encoding='latin1'))
        except EOFError:
            break
#
# print(objects[0]['test_videos'])
trainlist = objects[0]['test_videos'][0]
row = []

# /home/deeperAction22/ActionRec/sin/HIT_text_ecoder_without_RGB/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv
# /home/deeperAction22/ActionRec/webber/image_and_text_encoder_with_prompt/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv
with open("/work/sc19981018/sin/new/ablation/ucf_occulation/data/output/dense_serial_debug/inference/ava_video_val_v2.2/result_ucf.csv", newline='') as csvfile:
    rows = csv.reader(csvfile)
    
    label_indexes = []
    data = []
    for i in tqdm(rows):
        if float(i[-1]) > 0: # score threshold
            video_index = trainlist.index(i[0])
            video_type = i[0].split('/')[0] # video sport type
            video_label_index = labels.index(video_type)
            frame_number = int(i[1])
            x1, y1, x2, y2 = (i[2], i[3], i[4], i[5])
            our_label_index = int(i[6]) - 1
            ori_label_index = labels.index(our_test_labels[our_label_index])
            score = float(i[7])
        
            item = np.array([video_index, frame_number, ori_label_index, score, x1, y1, x2, y2, video_label_index])
            data.append(item)
            label_indexes.append(ori_label_index)

new_item = []
for step in range(0, len(data), len(our_test_labels)):
    group = np.array(data[step:step+6])[:, 3]
    group = np.argsort(group)
    new_item.append(
        data[step + group[-1]]
    )

print(set(sorted(label_indexes)))
new_item = np.array(new_item, dtype='float32')  
data = np.array(data, dtype='float32')      
with open('frame_detections.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=2)

print("Done")