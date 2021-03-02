# VQA Collection

PyTorch reimplementation of VQA models, including:

- [x] [Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering, CVPR 2018](https://arxiv.org/abs/1707.07998)
- [ ] [VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions](https://arxiv.org/abs/1803.07464)
  - [ ] Caption selection strategy (should be added into preprocessing)
- [ ] [Generating Question Relevant Captions to Aid Visual Question Answering](https://arxiv.org/abs/1906.00513)
  - [x] Caption embedding module
  - [ ] Caption selection strategy
- [ ] [Relation-Aware Graph Attention Network for Visual Question Answering](https://arxiv.org/abs/1903.12314)
  - [x] Spatial relation
  - [ ] Semantic relation
  - [ ] Implicit relation
- [ ] [Exploring Visual Relationship for Image Captioning, ECCV 2018](https://arxiv.org/abs/1809.07041)


## Comparison



| Model | VQA | Captioning | Gragh network | multi-task |
|:-:|:-:|:-:|:-:|:-:|
| Up-Down | v | v |  | x |
| VQA-E | v | v |  | use captions that is most similar to corresponding Q-A pairs |
| Q-Relevant | v | v |  | use all captions but only back-propagate from the most relevant one |
| ReGAT | v | | v | x |
| GCN-LSTM | | v | v | x |


## Preprocessing

### Visual Feature Extraction


### VQA and COCO Caption Dataset Alignment


## Results


| Exp | Model | Yes/No | Number | Other | All |
| - | :-: | :-: | :-: | :-: | :-: |
| exp1 | base | 77.25 | 39.08 | 43.96 | 55.83 |
| exp2 | new  | 78.14 | 40.66 | 44.89 | 56.83 |


## References

- Caption generation
    - https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    - https://github.com/poojahira/image-captioning-bottom-up-top-down
- 2017 VQA Challenge
    - https://github.com/hengyuan-hu/bottom-up-attention-vqa
- Graph convolution network
    - https://github.com/tkipf/pygcn
    - https://github.com/meliketoy/graph-cnn.pytorch
- Visualizing attention map
    - https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
