import numpy as np

import torch
import torch.nn as nn

from util.modules import FCNet, CaptionEmbedding, LReLUNet
from util.attention import ConcatAttention


class BasePredictor(nn.Module):
    """
    This is for the winning entry of the 2017 VQA Challenge.
    """
    def __init__(self,
                 v_dim: int,
                 hidden_dim: int,
                 ans_dim: int,
                 device: str,
                 cls_layer: int = 2,
                 dropout: float = 0.5,
    ):
        super().__init__()

        # Non-linear layers for image features
        self.q_net = FCNet(hidden_dim, hidden_dim)

        # Non-linear layers for question
        self.v_net = FCNet(v_dim, hidden_dim)

        # Classifier
        self.classifier = FCNet(
            in_dim=hidden_dim,
            mid_dim=2*hidden_dim,
            out_dim=ans_dim,
            layer=cls_layer, dropout=dropout
        )

    def forward(self, v, q):
        """Input:
            v: [batch, num_objs, v_dim]
            q: [batch, hidden_dim]
        """
        v = v.sum(1) # [batch, v_dim]

        # FC layers
        q = self.q_net(q) # [batch, hidden_dim]
        v = self.v_net(v) # [batch, hidden_dim]
        
        # Fuse visual and question features (multiplication here)
        joint = q * v # [batch, hidden_dim]
        
        return self.classifier(joint)


class PredictorwithCaption(nn.Module):
    """
    This is for the system described in 'Generating Question Relevant Captions to Aid Visual Question Answering' (https://arxiv.org/abs/1906.00513)
    """
    def __init__(self,
                    v_dim: int,
                    hidden_dim: int,
                    ans_dim: int,
                    device: str,
                    neg_slope: float = 0.01,
                    cls_layer: int = 2,
                    dropout: float = 0.5,
    ):
        super().__init__()
        
        # TODO

        # Classifier
        self.classifier = nn.Sequential(
            LReLUNet(hidden_dim, ans_dim, neg_slope),
            nn.Sigmoid()
        )

    def forward(self, v, w):
        """Input:
            v: [batch, num_objs, hidden_dim]
            w: (q, c)
                q: [batch, hidden_dim]
        """
        q, c = w
        del w

        # TODO

        return