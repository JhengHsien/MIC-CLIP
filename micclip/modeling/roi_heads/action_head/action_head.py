import torch

from .roi_action_feature_extractor import make_roi_action_feature_extractor
from .roi_action_predictors import make_roi_action_predictor
from .text_feature_gen import make_text_feature_generator
from .inference import make_roi_action_post_processor
from .loss import make_roi_action_loss_evaluator
from .metric import make_roi_action_accuracy_evaluator
from micclip.modeling.utils import prepare_pooled_feature
from micclip.utils.comm import all_reduce

from micclip.structures.bounding_box import BoxList
from .prompt import make_video_prompt
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import time
from .motion_transformer._3d_motion import motion_3d 

class ROIActionHead(torch.nn.Module):
    """
    Generic Action Head class.
    """

    def __init__(self, cfg, actionlist, actiondict, actiontoken, device):
        super(ROIActionHead, self).__init__()
        self.cfg = cfg
        self.feature_extractor = make_roi_action_feature_extractor(cfg)

        self.text_feature_generator = make_text_feature_generator(cfg, actionlist, actiondict, actiontoken, device)
        self.post_processor = make_roi_action_post_processor(cfg)
        self.loss_evaluator = make_roi_action_loss_evaluator(cfg)
        self.accuracy_evaluator = make_roi_action_accuracy_evaluator(cfg)
        self.test_ext = cfg.TEST.EXTEND_SCALE

        self.prompts_generator = make_video_prompt(cfg)
        self.fc1 = nn.Linear(8,11)
        self.device = device

        # motion
        self.motion_model = motion_3d(cfg, device)
        self.pool = nn.AvgPool1d(2, stride=2)
        self.MHA = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.MLP = nn.Sequential(
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Linear(16, 512),
            nn.Sigmoid()
        )
        
    def forward(self, boxes, objects=None, extras={}, part_forward=-1):
        # In training stage, boxes are from gt.
        # In testing stage, boxes are detected by human detector and proposals should be
        # enlarged boxes.
        assert not (self.training and part_forward >= 0)

        # extras contains what?
        # when train(part_forward == -1) , extras keys : person_pool,movie_ids,timestamps,cur_loss
        # when test stage 2(part_forward == 1) , extras keys : person_pool,movie_ids,timestamps,current_feat_p,current_feat_o,current_feat_h,current_feat_pose

        # webber
        # take image feature from CLIP image encoder
        # only for train and test stage 2

        # proposals,objects,keypoints contains 8 boxlists , each corresponds to a frame
        test_person_box = []
        if part_forward == 1:
            boxes = extras["current_feat_p"]
            test_person_box.append(boxes)
            objects = extras["current_feat_o"]

        if self.training:
            proposals = self.loss_evaluator.sample_box(boxes)
        else:
            proposals = boxes

        movieID = []
        time_stamp = []
        captions = []
        if part_forward != 1:
            image_paths = []
            for box in proposals:
                image_root = "data/jhmdb/videos/"
                movie_id = box.get_field('movie_id')
                timestamp = box.get_field('timestamp')             
                str_timestamp = str(timestamp).zfill(5)
                image_root = image_root + movie_id + '/' + str_timestamp + '.png'
                image_paths.append(image_root)
                movieID.append(movie_id)
                time_stamp.append(timestamp)
        else:
            image_paths = []
            for i in range(len(extras['movie_ids'])):
               image_root = "data/jhmdb/videos/"
               movie_id = extras['movie_ids'][i]
               timestamp = extras['timestamps'][i]
               str_timestamp = str(timestamp).zfill(5)
               image_root = image_root + movie_id + '/' + str_timestamp + '.png'
               movieID.append(movie_id)
               time_stamp.append(timestamp)
               image_paths.append(image_root)

            with open("./test_bbox.txt", "a") as f:
                for i, person in enumerate(test_person_box):
                    print(image_paths[i], ' | ', person, '\n', file=f)
        
        # get label text feature and image feature, also take person feature,object feature
        tFeature, iFeature, person_feature, object_feature, prompt_iFeature, person_bbox = self.text_feature_generator(image_paths, proposals, objects)
        motions = self.motion_model.get_embedding(movieID, time_stamp, person_bbox)

        if part_forward != 0:
           x, x_pooled, x_objects, x_scene = self.feature_extractor(iFeature, person_feature, object_feature,
        proposals, objects, extras, part_forward)

        else:
            pooled_feature = prepare_pooled_feature(person_feature, boxes)
            if object_feature is None:
                object_pooled_feature = None
            else:
                object_pooled_feature = prepare_pooled_feature(object_feature, objects)

            return [pooled_feature, object_pooled_feature, None, None], {}, {}, {}

        # if part_forward != 1:
        b = x.shape[0]
        tFeature = tFeature.unsqueeze(0).expand(b, -1, -1)

        mean_image_text = torch.einsum("bd,bkd->bk", x, tFeature) # (8,15)
        indices = torch.argmax(mean_image_text, dim=1)
        mid_features = []
        for i in range(len(indices)):
            mid_features.append(tFeature[i][indices[i]].cpu().detach().numpy())
        mid_features = torch.tensor(mid_features).to(self.device)
        mid_features = (x + mid_features) / 2
        
        q = x.unsqueeze(0)
        k = motions.unsqueeze(0)
        v = motions.unsqueeze(0)
        x = self.MHA(q,k,v)[0]
        x = x.squeeze(0)

        x = torch.div(x,2)
        prompt_iFeature = torch.div(prompt_iFeature,2)
        x = torch.add(x, prompt_iFeature)
        
        
        text_features = tFeature + self.prompts_generator(tFeature, motions)

        motions = motions / motions.norm(dim=-1, keepdim=True)
        
        x = x / x.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        action_logits = torch.einsum("bd,bkd->bk", motions, tFeature)
        action_logits = action_logits / 0.07
        
        # get result confident score
        if not self.training:
            result = self.post_processor((action_logits,), boxes)
            return result, {}, {}, {}

        box_num = action_logits.size(0)
        box_num = torch.as_tensor([box_num], dtype=torch.float32, device=action_logits.device)
        all_reduce(box_num, average=True)

        loss_dict, loss_weight = self.loss_evaluator(
            [action_logits], mid_features, box_num.item(), motions
        )

        metric_dict = self.accuracy_evaluator(
            [action_logits], proposals, box_num.item(),
        )

        pooled_feature = prepare_pooled_feature(x_pooled, proposals)
        if x_objects is None:
            object_pooled_feature = []
        else:
            object_pooled_feature = prepare_pooled_feature(x_objects, objects)
            

        return (
            [pooled_feature, object_pooled_feature, None, None],
            loss_dict,
            loss_weight,
            metric_dict,
        )

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map

    def video_features_reduce(self, video_features, proposals):
        features = []

        for i in range(len(proposals)):

            box_num = len(proposals[i])
            for _ in range(box_num):
                features.append(video_features[i].cpu().detach().numpy())
            
        video_input = torch.tensor(np.stack(features)).to(self.device)
        return video_input


def build_roi_action_head(cfg, actionlist, actiondict, actiontoken, device):
    return ROIActionHead(cfg, actionlist, actiondict, actiontoken, device)
