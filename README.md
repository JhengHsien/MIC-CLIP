# HIT


This project is the official implementation of paper 
[Holistic Interaction Transformer Network for Action Detection](https://arxiv.org/abs/2210.12686) (**WACV 2023**), authored
by Gueter Josmy Faure, Min-Hung Chen and Shang-Hong Lai. 


## Demo Video

Coming Soon

## Installation 

You need first to install this project, please check [INSTALL.md](INSTALL.md)

## Data Preparation

To do training or inference on J-HMDB, please check [DATA.md](DATA.md)
for data preparation instructions. Instructions for other datasets coming soon.

## Model Zoo

Please see [MODEL_ZOO.md](MODEL_ZOO.md) for downloading models.

## Training and Inference

To do training or inference with AlphAction, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## Demo Program

To run the demo program on video or webcam, please check the folder [demo](demo).
We select 15 common categories from the 80 action categories of AVA, and 
provide a practical model which achieves high accuracy (about 70 mAP) on these categories. 

## Acknowledgement
We thankfully acknowledge the computing resource support of Huawei Corporation
for this project. 

## Citation

If this project helps you in your research or project, please cite
this paper:

```
@inproceedings{tang2020asynchronous,
  title={Asynchronous Interaction Aggregation for Action Detection},
  author={Tang, Jiajun and Xia, Jin and Mu, Xinzhi and Pang, Bo and Lu, Cewu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2020}
}
```