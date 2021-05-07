import numpy as np

import torch
import torch.nn as nn

from .modules import FCNet, CaptionEmbedding, LReLUNet, SentenceEmbedding
from .attention import ConcatAttention

def set_predictor(predictor_type: str,
                  v_dim: int,
                  embed_dim: int,
                  hidden_dim: int,
                  ans_dim: int,
                  device: str,
                  cls_layer: int,
                  dropout: float,
                  c_len: int,
                  neg_slope: float,
    ):
    if predictor_type == 'base':
        return BasePredictor(
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout
        ).to(device)

    if predictor_type == 'base-cap':
        return BaseCaptionPredictor(
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout
        ).to(device)
    
    if predictor_type == 'q-cap':
        return PredictorwithCaption(
            embed_dim=embed_dim,
            c_len=c_len,
            v_dim=v_dim,
            hidden_dim=hidden_dim,
            ans_dim=ans_dim,
            device=device,
            cls_layer=cls_layer,
            dropout=dropout,
            neg_slope=neg_slope
        ).to(device)

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
        self.device = device

        # Non-linear layers for question
        self.v_net = FCNet(v_dim, hidden_dim)

        # Classifier
        self.classifier = FCNet(
            in_dim=hidden_dim,
            mid_dim=2*hidden_dim,
            out_dim=ans_dim,
            layer=cls_layer,
            dropout=dropout
        )

    def forward(self, batch):
        v = batch['v'].to(self.device)
        q = batch['q'].to(self.device)

        v = v.sum(1) # [batch, v_dim]

        # FC layers
        v = self.v_net(v) # [batch, hidden_dim]
        
        # Joint question features (multiply)
        joint = q * v # [batch, hidden_dim]
        
        return self.classifier(joint)


class BaseCaptionPredictor(BasePredictor):
    def __init__(self,
                 v_dim: int,
                 embed_dim: int,
                 hidden_dim: int,
                 ans_dim: int,
                 device: str,
                 cls_layer: int = 2,
                 dropout: float = 0.5,
    ):
        super().__init__(v_dim, hidden_dim, ans_dim, device, cls_layer, dropout)
        self.c_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=1,
            device=device,
            rnn_type='GRU'
        )
        self.c_net = FCNet(hidden_dim, hidden_dim, dropout=dropout)
    
    def forward(self, batch):
        v = batch['v'].to(self.device)
        q = batch['q'].to(self.device)
        c = batch['c'].to(self.device)

        # v_mean
        v = v.sum(1) # [batch, v_dim]

        # caption embedding
        c = self.c_net(self.c_rnn(c))

        # FC layers
        v = self.v_net(v) # [batch, hidden_dim]

        # Joint visual and caption embedding (add)
        joint = c + v
        
        # Joint question features (multiply)
        joint = q * joint # [batch, hidden_dim]
        
        return self.classifier(joint)



class PredictorwithCaption(nn.Module):
    """
    This is for the system described in 'Generating Question Relevant Captions to Aid Visual Question Answering' (https://arxiv.org/abs/1906.00513)
    """
    def __init__(self,
                    embed_dim: int,
                    c_len: int,
                    v_dim: int,
                    hidden_dim: int,
                    ans_dim: int,
                    device: str,
                    cls_layer: int = 2,
                    dropout: float = 0.5,
                    neg_slope: float = 0.01,
    ):
        super().__init__()
        self.device = device
        
        # Caption embedding module
        self.v_net = LReLUNet(v_dim, hidden_dim, neg_slope)
        self.caption_embedding = CaptionEmbedding(
            v_dim=hidden_dim,
            q_dim=hidden_dim,
            c_dim=embed_dim, 
            hidden_dim=hidden_dim,
            max_len=c_len,
            device=device,
            dropout=dropout
        )
        self.c_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        
        # For caption-attended visual features
        self.vq_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.joint_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.vqc_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)

        # Classifier
        self.classifier = nn.Sequential(
            LReLUNet(hidden_dim, ans_dim, neg_slope),
            nn.Sigmoid()
        )

    def forward(self, batch):
        for i in batch: batch[i] = batch[i].to(self.device)
        batch['v'] = self.v_net(batch['v'])

        # Caption embedding
        v = batch['v'].sum(1) # [batch, v_dim]
        c = self.caption_embedding(v, batch['q'], batch['c']) # [batch, hidden_dim]
        # c = batch['c'].to(self.device).sum(1) # [batch, hidden_dim]

        # Produce caption-attended visual features
        v = self.vq_net(v) # [batch, hidden_dim]
        c = self.c_net(c) # [batch, hidden_dim]
        joint = self.joint_net(c * v)
        joint = nn.functional.softmax(joint, 1) # [batch, num_objs, hidden_dim]
        v = (joint.unsqueeze(1).repeat(1, batch['v'].size(1), 1) * batch['v']).sum(1) # [batch, hidden_dim]

        # To better incorporate the information from the captions into the VQA process,
        # add the caption feature ot the attended image features,
        # and then element-wise multiply by the question features.
        v = self.vqc_net(v) # [batch, hidden_dim]
        joint = batch['q'] * (v + c) # [batch, hidden_dim]

        return self.classifier(joint) # [batch, ans_dim]