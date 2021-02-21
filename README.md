# VQA Collection

PyTorch reimplementation of VQA models, including:

1. [Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering, CVPR 2018](https://arxiv.org/abs/1707.07998)
2. [VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions](https://arxiv.org/abs/1803.07464)
  - [ ] Caption selection strategy (should be added into preprocessing)
3. [Generating Question Relevant Captions to Aid Visual Question Answering](https://arxiv.org/abs/1906.00513)
  - [x] Caption embedding module
  - [ ] Caption selection strategy
4. [Relation-Aware Graph Attention Network for Visual Question Answering](https://arxiv.org/abs/1903.12314)
  - [x] Spatial relation
  - [ ] Semantic relation
  - [ ] Implicit relation
5. [Exploring Visual Relationship for Image Captioning, ECCV 2018](https://arxiv.org/abs/1809.07041)


## Comparison



| Model | VQA | Captioning | Gragh network | multi-task |
|:-:|:-:|:-:|:-:|:-:|
| Up-Down | v | v |  | x |
| VQA-E | v | v |  | use captions relevant to corresponding Q-A pairs  |
| Q-Relevant | v | v |  | only back-propagate from the most relevant captions |
| ReGAT | v | | v | x |
| GCN-LSTM | | v | v | x |


## Preprocessing

### Visual Feature Extraction


### VQA and COCO Caption Dataset Alignment


## Results


| Exp | Model | All | Yes/No | Number | Other |
| - | :-: | :-: | :-: | :-: | :-: |
| exp1 | Up-Down |||||

## References

Codes for caption generation adapted from

- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- https://github.com/poojahira/image-captioning-bottom-up-top-down

Codes for the 2017 VQA Challenge adapted from

- https://github.com/hengyuan-hu/bottom-up-attention-vqa

Codes for graph convolution network adapted from
- https://github.com/tkipf/pygcn
- https://github.com/meliketoy/graph-cnn.pytorch
