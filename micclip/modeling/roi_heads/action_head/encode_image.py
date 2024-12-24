from PIL import Image
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import sys
from .cct import CrossFrameCommunicationTransformer
from clip import clip
import torchvision.transforms as transfrom

device = torch.device("cuda")
image_resolution = 7200
vision_patch_size = 2
vision_patch_size = 2
vision_width = 320
vision_heads = vision_width // 64
vision_layers = 1
visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=512
        )

def encode_image(image):
    # x = visual(image)
    print('encode_image')
    # return x 

def encode_video(image_list):
    images = []
    video_features_list = []
    # transformer = transfrom.Compose([transfrom.ToTensor()])
        
    for i in range(len(image_list)):
        # path = image_paths[i]
        # image = Image.open(path).convert("RGB")
        
        #image = transformer(image)
        # preprocess = clip.load('ViT-B/16', device=device, jit=False, return_intermediate_text_feature=0)
        
        # image = preprocess(image)
        #images.append(image)
    
        #print(image)
        image = image_list[i]
        c,h,w = image.size()
        image = image.reshape(-1,c,h,w)

        cls_features, img_features = encode_image(image)
        # img_features = self.prompts_visual_ln(img_features)
        # img_features = img_features @ self.prompts_visual_proj
        
        cls_features = cls_features.view(b, t, -1)
        img_features = img_features.view(b,t,-1,cls_features.shape[-1])
        
        video_features = self.mit(cls_features)
        video_features_list.append(video_features)
    

    return video_features_list


        