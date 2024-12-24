import torch
from torch import nn
from torch.nn import functional as F

from micclip.modeling import registry
from micclip.modeling.poolers import make_3d_pooler
from micclip.modeling.roi_heads.action_head.hit_structure import make_hit_structure
from micclip.modeling.utils import cat, pad_sequence, prepare_pooled_feature
from micclip.utils.IA_helper import has_object, has_hand
from micclip.structures.bounding_box import BoxList
from micclip.modeling.roi_heads.action_head.text_feature_gen import CLIPencoder
import time
import os


@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD

        self.pooler = make_3d_pooler(head_cfg)

        resolution = head_cfg.POOLER_RESOLUTION

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))
        self.dim_in = config.MODEL.HIT_STRUCTURE.DIM_INNER
        if config.MODEL.HIT_STRUCTURE.ACTIVE:
            self.max_feature_len_per_sec = config.MODEL.HIT_STRUCTURE.MAX_PER_SEC
            self.hit_structure = make_hit_structure(config, self.dim_in)

        representation_size = head_cfg.MLP_HEAD_DIM
        # print('MLP_2 size:',representation_size) # webber : MLP2 size, init = 1024, set to 512

        fc1_dim_in = self.dim_in
        if config.MODEL.HIT_STRUCTURE.ACTIVE and (config.MODEL.HIT_STRUCTURE.FUSION == "concat"):
            fc1_dim_in += config.MODEL.HIT_STRUCTURE.DIM_OUT

        self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc3 = nn.Linear(11, 8)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        self.dim_out = representation_size
        self.clip = CLIPencoder(self.config,None,None,None,torch.device("cuda")) # webber : memory take image feature
        self.image_feature_pool = {} # webber : memory take image feature

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_x = slow_x.mean(dim=2, keepdim=True)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_x = fast_x.mean(dim=2, keepdim=True)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def max_pooling_zero_safe(self, x):
        if x.size(0) == 0:
            _, c, t, h, w = x.size()
            res = self.config.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION
            x = torch.zeros((0, c, 1, h - res + 1, w - res + 1), device=x.device)
        else:
            x = self.max_pooler(x)
        return x

    def forward(self, image_feature, person_feature, object_feature, proposals, objects=None, extras={}, part_forward=-1):
        ia_active = hasattr(self, "hit_structure")
        if part_forward == 1:
            person_pooled = cat([box.get_field("pooled_feature") for box in proposals])
            if objects is None:
                object_pooled = None
            else:
                object_pooled = cat([box.get_field("pooled_feature") for box in objects])

            scene_pooled = image_feature

        else:
            person_pooled = person_feature
            object_pooled = object_feature
            scene_pooled = image_feature

        
        if object_pooled.shape[0] == 0:
            object_pooled = None

        x_after = person_pooled
        scene_pooled = scene_pooled.unsqueeze(1)
        if ia_active:
            tsfmr = self.hit_structure
            mem_len = self.config.MODEL.HIT_STRUCTURE.LENGTH
            mem_rate = self.config.MODEL.HIT_STRUCTURE.MEMORY_RATE
            use_penalty = self.config.MODEL.HIT_STRUCTURE.PENALTY
            # webber : memory take image feature
            memory_person = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                                                       self.max_feature_len_per_sec, tsfmr.dim_others,
                                                                       person_pooled, proposals, use_penalty)
            # ic(memory_person)
            # RGB stream
            ia_feature = self.hit_structure(person_pooled, proposals, object_pooled, objects, scene_pooled, memory_person, None, phase="rgb")
            # x_after = self.fusion(x_after, ia_feature, self.config.MODEL.HIT_STRUCTURE.FUSION)
            x_after = ia_feature
        x_after = x_after.view(x_after.size(0), -1)

        return x_after, person_pooled, object_pooled, scene_pooled

    # can use Q-transformer
    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, fixed_dim, current_x, current_box, use_penalty):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        if use_penalty and self.training:
            cur_loss = extras["cur_loss"]
        else:
            cur_loss = 0.0
        current_feat = prepare_pooled_feature(current_x, current_box, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            movie_len = len([name for name in os.listdir('data/jhmdb/videos/' + movie_id) if os.path.isfile(os.path.join('data/jhmdb/videos/' + movie_id, name))])
            length = movie_len // sum(mem_len) # 180 / 16


            total_frames = movie_len
            mem_before = mem_len[0]
            mem_after = mem_len[1]
            total_needed_frames = mem_before + mem_after
            all_frames = list(range(1, total_frames + 1))

            # Step 1: Calculate how many frames we can take after the timestamp
            remaining_after_frames = total_frames - timestamp

            # If we don't have enough frames after the timestamp, adjust mem_before
            if remaining_after_frames < mem_after:
                mem_before += (mem_after - remaining_after_frames)
                mem_after = remaining_after_frames

            # If we don't have enough frames before the timestamp, adjust mem_after
            if timestamp < mem_before:
                mem_after += (mem_before - timestamp)
                mem_before = timestamp

            # Step 2: Select frames before the timestamp
            if mem_before > 0:
                before_step = timestamp // mem_before
                before_frames = [i * before_step for i in range(1, mem_before + 1)]
            else:
                before_frames = []

            # Step 3: Select frames after the timestamp
            if mem_after > 0:
                after_step = remaining_after_frames // mem_after
                after_frames = [timestamp + (i * after_step -1) for i in range(1, mem_after+1)]
            else:
                after_frames = []

            # print(before_frames, after_frames)
            cache_cur_mov = feature_pool[movie_id]
            
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, movie_id, mem_ind, max_boxes, cur_loss, use_penalty)
                                for mem_ind in before_frames]
        
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, movie_id, mem_ind, max_boxes, cur_loss, use_penalty)
                                for mem_ind in after_frames]
            mem_box_list = mem_box_list_before + mem_box_list_after # memory take image feature
            mem_feature_list += [image_feature for image_feature in mem_box_list]

        seq_length = sum(mem_len) # memory take image feature
        person_per_seq = len(mem_feature_list)
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, 512, 1, 1, 1)
        mem_feature = mem_feature.to(device)

        return mem_feature

    def check_fetch_mem_feature(self, movie_cache, movie_id, mem_ind, max_num, cur_loss, use_penalty):
        # webber : memory take image feature
        image_path = []
        image_root = "data/jhmdb/videos/"
        str_timestamp = str(mem_ind).zfill(5)
        # check whether this image feature exists
        key = movie_id + '/' + str_timestamp
        # if key not in self.image_feature_pool:
        image_root = image_root + movie_id + '/' + str_timestamp + '.png'
        image_path.append(image_root)
        image_feature = self.clip.generate_memory_image_feature(image_path)
        self.image_feature_pool[key] = image_feature

        if mem_ind not in movie_cache:
            return image_feature
        box_list = movie_cache[mem_ind]
        box_list = self.sample_mem_feature(box_list, max_num)
        
        box_list.add_field("image_feature", self.image_feature_pool[key])
        ######################################

        if use_penalty and self.training:
            loss_tag = box_list.delete_field("loss_tag")
            penalty = loss_tag / cur_loss if loss_tag < cur_loss else cur_loss / loss_tag
            features = box_list.get_field("pooled_feature") * penalty
            box_list.add_field("pooled_feature", features)
        # return box_list
        return image_feature

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError


def make_roi_action_feature_extractor(cfg):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
