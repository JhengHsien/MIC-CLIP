#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import csv
import numpy as np
from tqdm import tqdm

objects = []

with (open("JHMDB-GT.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile,encoding='latin1'))
        except EOFError:
            break
print(objects[0].keys())
trainlist = objects[0]['test_videos'][0]

row = []

# /home/deeperAction22/ActionRec/sin/HIT_text_encoder_without_RGB/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv
# /home/deeperAction22/ActionRec/AIA_github_w/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv
with open("/work/sc19981018/sin/new/IVP_MOTION_UCF_JHMDB/data/output/dense_serial_debug/inference/ava_video_val_v2.2/result_ucf.csv", newline='') as csvfile:
    rows = csv.reader(csvfile)
    
    label_indexes = []
    data = []
    for i in tqdm(rows):
        if float(i[-1]) > 0: # score threshold
            video_index = trainlist.index(i[0])
            video_type = i[0].split('/')[0] # video sport type
            frame_number = int(i[1])
            x1, y1, x2, y2 = (i[2], i[3], i[4], i[5])
            #our_label_index = int(i[6]) - 1
            #ori_label_index = labels.index(our_test_labels[our_label_index])
            ori_label_index = int(i[6]) - 1
            score = float(i[7])
        
            item = np.array([video_index, frame_number, ori_label_index, score, x1, y1, x2, y2])
            data.append(item)
            label_indexes.append(ori_label_index)

print(set(sorted(label_indexes)))
data = np.array(data, dtype='float32')      
with open('frame_detections.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=2)

print("Done")



