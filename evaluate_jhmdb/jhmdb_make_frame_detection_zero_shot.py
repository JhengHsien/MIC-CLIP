#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import csv
import numpy as np
from tqdm import tqdm


# In[4]:


labels = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw', 'walk', 'wave']
# test_label_file = open("/work/sc19981018/jhmdb_label_split/2/test_label.txt", "r")
test_label_file = open("jhmdb_label_split/1/test_label.txt", "r")
data = test_label_file.read()
our_test_labels = data.split("\n")
our_test_labels = our_test_labels[:-1]
test_label_file.close()

# our_test_labels = ['clap','sit','wave','throw','pullup','catch']

objects = []

with (open("evaluate_jhmdb/JHMDB-GT.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile,encoding='latin1'))
        except EOFError:
            break
# print(objects[0].keys())
# print(objects[0]['test_videos'][0])
trainlist = objects[0]['test_videos'][0]
row = []

# /home/deeperAction22/ActionRec/sin/HIT_text_ecoder_without_RGB/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv
# /home/deeperAction22/ActionRec/webber/image_and_text_encoder_with_prompt/data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv
with open("data/output/dense_serial_debug/inference/jhmdb_val/result_jhmdb.csv", newline='') as csvfile:
    rows = csv.reader(csvfile)
    label_indexes = []
    data = []

    # total = 0
    # for _, row in enumerate(rows):
    #     total += 1

    for i in tqdm(rows):
        # for step in range(0, total, len(our_test_labels)):
        #     if ()

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