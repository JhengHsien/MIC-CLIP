import os
import torch.utils.data as data
import time
import torch
import numpy as np
from micclip.structures.bounding_box import BoxList
from collections import defaultdict
from micclip.utils.video_decode import av_decode_video
import re
from PIL import Image

import json
import numpy as np
from micclip.config import cfg

from micclip.dataset.datasets.iou_calculator import iou

# This is used to avoid pytorch issuse #13246
class NpInfoDict(object):
    def __init__(self, info_dict, key_type=None, value_type=None):
        keys = sorted(list(info_dict.keys()))
        self.key_arr = np.array(keys, dtype=key_type)
        self.val_arr = np.array([info_dict[k] for k in keys], dtype=value_type)
        # should not be used in dataset __getitem__
        self._key_idx_map = {k:i for i,k in enumerate(keys)}
    def __getitem__(self, idx):
        return self.key_arr[idx], self.val_arr[idx]
    def __len__(self):
        return len(self.key_arr)
    def convert_key(self, org_key):
        # convert relevant variable whose original value is in the key set.
        # should not be used in __getitem__
        return self._key_idx_map[org_key]

# This is used to avoid pytorch issuse #13246
class NpBoxDict(object):
    def __init__(self, id_to_box_dict, key_list=None, value_types=[]):
        value_fields, value_types = list(zip(*value_types))

        # if "keypoints" not in value_fields:
        assert 'bbox' in value_fields

        if key_list is None:
            key_list = sorted(list(id_to_box_dict.keys()))
        self.length = len(key_list)

        pointer_list = []
        value_lists = {field: [] for field in value_fields}
        cur = 0
        pointer_list.append(cur)
        for k in key_list:
            box_infos = id_to_box_dict[k]
            cur += len(box_infos)
            pointer_list.append(cur)
            for box_info in box_infos:
                for field in value_fields:
                    value_lists[field].append(box_info[field])
        self.pointer_arr = np.array(pointer_list, dtype=np.int32)
        self.attr_names = np.array(["vfield_" + field for field in value_fields])
        for field_name, value_type, attr_name in zip(value_fields, value_types, self.attr_names):
            setattr(self, attr_name, np.array(value_lists[field_name], dtype=value_type))

    def __getitem__(self, idx):
        l_pointer = self.pointer_arr[idx]
        r_pointer = self.pointer_arr[idx + 1]
        ret_val = [getattr(self, attr_name)[l_pointer:r_pointer] for attr_name in self.attr_names]
        return ret_val

    def __len__(self):
        return self.length

class DatasetEngine(data.Dataset):
    def __init__(self, video_root, ann_file, remove_clips_without_annotations, frame_span, box_file=None, eval_file_paths={},
                 box_thresh=0.0, action_thresh=0.0, transforms=None, object_file=None, object_transforms=None, keypoints_file=None,):

        print('loading annotations into memory...')
        tic = time.time()
        self.json_dict = json.load(open(ann_file, 'r'))
        assert type(self.json_dict) == dict, 'annotation file format {} not supported'.format(type(self.json_dict))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        self.video_root = video_root
        self.transforms = transforms
        self.frame_span = frame_span

        # These two attributes are used during ava evaluation...
        # Maybe there is a better implementation
        self.eval_file_paths = eval_file_paths
        self.action_thresh = action_thresh

        clip2ann = defaultdict(list)

        
        if "annotations" in self.json_dict:
            for ann in self.json_dict["annotations"]:
                action_ids = ann["action_ids"]
                one_hot = np.zeros(22, dtype=np.bool)
                one_hot[action_ids] = True
                packed_act = one_hot[1:]
                clip2ann[ann["image_id"]].append(dict(bbox=ann["bbox"], packed_act=packed_act))

        movies_size = {}
        clips_info = {}
        train_video = []
        test_video = []
        # for train
        if(not box_file):
            f = open(cfg.DATASET_VIDEO, 'rb')
            for line in f.readlines():
                name = line[:-1].decode("utf-8") 
                train_video.append(name)
        # for test
        else:
            f = open(cfg.TESTSET_VIDEO, 'rb')
            for line in f.readlines():
                name = line[:-1].decode("utf-8") 
                test_video.append(name)
        
        for img in self.json_dict["images"]:
            mov = img["movie"]
            #################### zero shot #################################
            if (not box_file):
                if mov in train_video:
                    if mov not in movies_size:
                        movies_size[mov] = [img["width"], img["height"]]
                    clips_info[img["id"]] = [mov, img["timestamp"]]
            else:
                if mov in test_video:
                    if mov not in movies_size:
                        movies_size[mov] = [img["width"], img["height"]]
                    clips_info[img["id"]] = [mov, img["timestamp"]]

            #################### train label = test label #################################
            #if (not box_file):
            #    if mov not in movies_size:
            #        movies_size[mov] = [img["width"], img["height"]]
            #    clips_info[img["id"]] = [mov, img["timestamp"]]
            #else:
            #    if mov not in movies_size:
            #        movies_size[mov] = [img["width"], img["height"]]
            #    clips_info[img["id"]] = [mov, img["timestamp"]]
            ################################################################################
        
        for items in self.json_dict["annotations"]:
            image_ids = items["image_id"]
            if image_ids in clips_info:
                movie_name = clips_info[image_ids][0] + str(clips_info[image_ids][1]).zfill(5)

        self.movie_info = NpInfoDict(movies_size, value_type=np.int32)
        clip_ids = sorted(list(clips_info.keys()))

        if remove_clips_without_annotations:
            clip_ids = [clip_id for clip_id in clip_ids if clip_id in clip2ann]

        if box_file:
            # this is only for validation or testing
            # we use detected boxes, so remove clips without boxes detected.
            imgToBoxes = self.load_box_file(box_file, box_thresh)
            clip_ids = [
                img_id
                for img_id in clip_ids
                if len(imgToBoxes[img_id]) > 0
            ]
            self.det_persons = NpBoxDict(imgToBoxes, clip_ids,
                                         value_types=[("bbox", np.float32), ("score", np.float32)])
        else:
            self.det_persons = None

        if object_file:
            imgToObjects = self.load_box_file(object_file)
            self.det_objects = NpBoxDict(imgToObjects, clip_ids,
                                         value_types=[("bbox", np.float32), ("score", np.float32)])
        else:
            self.det_objects = None

        if object_transforms:
            self.object_transforms = object_transforms
        else:
            self.object_transforms = None

        self.anns = NpBoxDict(clip2ann, clip_ids, value_types=[("bbox", np.float32), ("packed_act", np.uint8)])

        clips_info = {
            clip_id:
                [
                    self.movie_info.convert_key(clips_info[clip_id][0]),
                    clips_info[clip_id][1]
                ] for clip_id in clip_ids
        }
        self.clips_info = NpInfoDict(clips_info, value_type=np.int32)


    def __getitem__(self, idx):
        _, clip_info = self.clips_info[idx]

        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        # video_data = self._decode_video_data(movie_id, timestamp)

        im_w, im_h = movie_size

        if self.det_persons is None:
            # Note: During training, we only use gt. Thus we should not provide box file,
            # otherwise we will use only box file instead.

            boxes, packed_act = self.anns[idx]

            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)  # guard against no boxes
            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xyxy")

            # Decode the packed bits from uint8 to one hot, since AVA has 80 classes,
            # it can be exactly denoted with 10 bytes, otherwise we may need to discard some bits.
            # one_hot_label = np.unpackbits(packed_act, axis=1)
            one_hot_label = torch.as_tensor(packed_act, dtype=torch.uint8)

            boxes.add_field("labels", one_hot_label)
        else:
            boxes, box_score = self.det_persons[idx]
            boxes_tensor = torch.as_tensor(boxes).reshape(-1, 4)

            # the highest person det score in this frame
            max_person_score = max(box_score)
            # the threshold we can tolerate
            thres = 0.001 # 0.001
            # remove after this idx, default = all person
            remove_idx = len(box_score)
            # we remove the bad detected person in this frame
            for i in range(len(box_score)):
                if box_score[i] < max_person_score - thres:
                    remove_idx = i
                    break
            boxes_tensor = boxes_tensor[:remove_idx,:]
            box_score = box_score[:remove_idx]

            boxes = BoxList(boxes_tensor, (im_w, im_h), mode="xyxy")
            boxes.add_field("det_score", torch.as_tensor(box_score))
            

        boxes.add_field("movie_id", movie_id)
        boxes.add_field("timestamp", timestamp)
        boxes = boxes.clip_to_image(remove_empty=True)

        # Get boxes before the transform
        # To calculate correct IoU
        orig_boxes = boxes.bbox
        # extra fields
        extras = {}

        if self.transforms is not None:

            objects = None
            if self.det_objects is not None:
                objects = self.get_objects(idx, im_w, im_h)
            keypoints = None
            # if self.det_keypoints is not None:
            #     keypoints = self.get_keypoints(idx, im_w, im_h, orig_boxes)

            extras["movie_id"] = movie_id
            extras["timestamp"] = timestamp

            # change to original box
            return boxes, objects, extras, idx

        return boxes, idx, movie_id, timestamp

    def return_null_box(self, im_w, im_h):
        return BoxList(torch.zeros((0, 4)), (im_w, im_h), mode="xyxy")

    def get_objects(self, idx, im_w, im_h):
        obj_boxes = self.return_null_box(im_w, im_h)
        if hasattr(self, 'det_objects'):
            boxes, box_score = self.det_objects[idx]

            if len(box_score) == 0:
                return obj_boxes
            obj_boxes_tensor = torch.as_tensor(boxes).reshape(-1, 4)
            obj_boxes = BoxList(obj_boxes_tensor, (im_w, im_h), mode="xyxy")
            # .convert("xyxy")

            scores = torch.as_tensor(box_score)
            obj_boxes.add_field("scores", scores)

        return obj_boxes
    
    def get_keypoints(self, idx, im_w, im_h, orig_boxes):
        kpts_boxes = self.return_null_box(im_w, im_h)
        keypoints, boxes, box_score = self.det_keypoints[idx]

        if len(box_score) == 0:
            kpts_boxes = BoxList(torch.zeros((orig_boxes.shape[0], 4)).reshape(-1, 4), (im_w, im_h), mode="xyxy")
            kpts_boxes.add_field("keypoints", np.zeros((orig_boxes.shape[0], 17, 3)))
            return kpts_boxes

        # Keep only the keypoints with corresponding boxes in the GT
        boxes = np.array(boxes)
        orig_boxes = orig_boxes.cpu().numpy()
        idx_to_keep = np.argmax(iou(orig_boxes, boxes), 1)
        boxes = boxes[idx_to_keep]
        keypoints = np.array(keypoints)
        keypoints = keypoints[idx_to_keep]

        keypoints_boxes_tensor = torch.as_tensor(boxes).reshape(-1, 4)
        kpts_boxes = BoxList(keypoints_boxes_tensor, (im_w, im_h), mode="xyxy")

        scores = torch.as_tensor(box_score)
        kpts_boxes.add_field("scores", scores)
        kpts_boxes.add_field("keypoints", keypoints)

        return kpts_boxes


    def get_video_info(self, index):
        _, clip_info = self.clips_info[index]
        # mov_id is the id in self.movie_info
        mov_id, timestamp = clip_info
        # movie_id is the human-readable youtube id.
        movie_id, movie_size = self.movie_info[mov_id]
        w, h = movie_size
        return dict(width=w, height=h, movie=movie_id, timestamp=timestamp)

    def load_box_file(self, box_file, score_thresh=0.0):
        import json

        print('Loading box file into memory...')
        tic = time.time()
        with open(box_file, "r") as f:
            box_results = json.load(f)
        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        boxImgIds = [box['image_id'] for box in box_results]

        imgToBoxes = defaultdict(list)
        for img_id, box in zip(boxImgIds, box_results):
            if float(box['score']) >= score_thresh:
                imgToBoxes[img_id].append(box)
        return imgToBoxes

    # def _decode_video_data(self, dirname, timestamp):
    #     # decode target video data from segment per second.
    #     video_folder = os.path.join(self.video_root, dirname)
    #     right_span = self.frame_span//2
    #     left_span = self.frame_span - right_span

    #     #load right
    #     cur_t = timestamp
    #     right_frames = []
    #     folder_list = np.array(os.listdir(video_folder))
    
    #     while cur_t < folder_list.shape[0]:
    #         if (cur_t - timestamp) > right_span:
    #             break
    #         video_path = os.path.join(video_folder, "{}.png".format(str(cur_t).zfill(5)))
    #         try:
    #             with Image.open(video_path) as img:
    #                 right_frames.append(img.convert('RGB'))
    #         except BaseException as e:
    #             raise RuntimeError('Caught "{}" when loading {}'.format(str(e), video_path))
    #         cur_t += 1
            
    #     #load left
    #     cur_t = timestamp-1
    #     left_frames = []
    #     while cur_t > 0:
    #         if (timestamp - cur_t) > left_span:
    #             break
    #         video_path = os.path.join(video_folder, "{}.png".format(str(cur_t).zfill(5)))
    #         # frames = cv2_decode_video(video_path)
    #         try:
    #             with Image.open(video_path) as img:
    #                 left_frames.append(img.convert('RGB'))
    #         except BaseException as e:
    #             raise RuntimeError('Caught "{}" when loading {}'.format(str(e), video_path))
    #         cur_t -= 1


    #     # min_frame_num = min(len(left_frames), len(right_frames))
    #     frames = left_frames + right_frames
    #     # frames = left_frames + right_frames
    #     video_data = np.stack(frames)
    #     # print(video_data.shape)
    #     return video_data


    def __len__(self):
        return len(self.clips_info)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Video Root Location: {}\n'.format(self.video_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str