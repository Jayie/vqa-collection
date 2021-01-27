# VQA Collection

PyTorch implementation of VQA models, including:

- [Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)
- (WIP) [Generating Question Relevant Captions to Aid Visual Question Answering](https://arxiv.org/abs/1906.00513)

  - VQA module Done
  - TODO: Caption module and loss function
- (TODO) [VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions](https://arxiv.org/abs/1803.07464)
- (TODO) [Relation-Aware Graph Attention Network for Visual Question Answering](https://arxiv.org/abs/1903.12314)


## Comparison



| Model | Caption generation | Gragh network | Use$L_{cap}$ |
| :-: | :-: | :-: | :-: |
| Up-Down |   |   |   |
| VQA-E | v |   |   |
| Q-Relevant | v |   | $L^c_i = - \sum_{t=1}^T \log(p(w^c_{i,t} \mid w^c_{i,t-1}))$, back-propagate the gradients only from the most relevant captions |
| ReGAT |   | v |   |


## Preprocessing

### Visual Feature Extraction


### VQA and COCO Caption Dataset Alignment


## Results


| Exp | Model | All | Yes/No | Number | Other |
| - | :-: | :-: | :-: | :-: | :-: |
| exp1 | Up-Down | (training) |   |   |   |

## References

Codes for caption generation adapted from

- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- https://github.com/poojahira/image-captioning-bottom-up-top-down

Codes for the 2017 VQA Challenge adapted from

- https://github.com/hengyuan-hu/bottom-up-attention-vqa
