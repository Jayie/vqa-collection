import numpy as np

import torch
import torch.nn as nn

from .gcn import GCN
from .modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding, CaptionEmbedding, LReLUNet
from .attention import ConcatAttention, MultiplyAttention


# This model is based on the winning entry of the 2017 VQA Challenge, following the system described in 
# 'Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering' (https://arxiv.org/abs/1707.07998) and 
# 'Tips and Tricks for Visual Question Answering: Learning from teh 2017 Challenge' (https://arxiv.org/abs/1708.02711)
#
# Code reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa

def set_att(att_type):
    return {
        'base': ConcatAttention,
        'new': MultiplyAttention
    }[att_type]

class BaseEncoder(nn.Module):
    """
    This is for the winning entry of the 2017 VQA Challenge.
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'GRU',
                 att_type: str = 'base'
    ):
        """Input:
            For question embedding:
                ntoken: number of tokens (i.e. size of vocabulary)
                embed_dim: dimension of question embedding
                hidden_dim: dimension of hidden layers
                rnn_layer: number of RNN layers
            For attention:
                v_dim: dimension of image features
                att_fc_dim: dimension of attention fc layer
            Others:
                device: device
                dropout: dropout (default = 0.5)
        """

        super().__init__()
        self.device = device

        # Word embedding for question
        self.embedding = nn.Embedding(ntoken+1, embed_dim, padding_idx=ntoken)
        
        # RNN for question
        self.q_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=rnn_layer,
            dropout=0.0,
            device=device,
            rnn_type=rnn_type
        )

        # Attention layer for image features based on questions
        self.attention = set_att(att_type)(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=att_fc_dim)

        # Non-linear layers for image features
        self.q_net = FCNet(hidden_dim, hidden_dim)

    def forward(self, batch):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
        Output:
            v: [batch, num_objs, v_dim]
            q: [batch, hidden_dim]
            att: [batch, num_objs, 1]
        """
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        
        # Embed words and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # Get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]

        # Get question-attended visual feature vq
        v = v_att * v # [batch, num_objs, v_dim]

        q = self.q_net(q) # [batch, hidden_dim]
        return v, q, v_att


class RelationEncoder(BaseEncoder):
    """
    This is for 'Relation-Aware Graph Network for Visual Question Answering'
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'GRU',
                 att_type: str = 'base',
                 conv_layer: int = 1,
                 conv_type: str = 'corr',
                 use_imp: bool = True,
                 use_spa: bool = False,
                 use_sem: bool = True,
                 num_objs: int = 36
    ):
        super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type, att_type)
        assert use_imp or use_spa or use_sem, 'Should use at least one relation'

        # Prepare GCN
        self.implicit_encoder = GCN(
            in_dim=v_dim,
            out_dim=v_dim,
            num_labels=12,
            device=device,
            conv_layer=conv_layer,
            conv_type=conv_type
        ) if use_imp else None

        self.spatial_encoder = GCN(
            in_dim=v_dim,
            out_dim=v_dim,
            num_labels=12,
            device=device,
            conv_layer=conv_layer,
            conv_type=conv_type
        ) if use_spa else None

        # Prepare fully-connected graph
        self.implicit_graph = (
            torch.ones(num_objs, num_objs) - torch.eye(num_objs)
        ).float().to(self.device)
        
    def forward(self, batch, graph_alpha=False):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
            graph: [batch, num_objs, num_objs]
        Output:
            v: [batch, num_objs, v_dim]
            q: [batch, hidden_dim]
            att: [batch, num_objs, 1]
        """
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        
        # Embed words and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # Get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]

        # Get question-attended visual feature vq
        v = v_att * v # [batch, num_objs, v_dim]

        q = self.q_net(q) # [batch, hidden_dim]

        # Get relation-aware visual feature
        new_v = torch.zeros_like(v)

        # Implicit graph
        if self.implicit_encoder:
            # graph = torch.ones_like(batch['graph']) - torch.eye(batch['graph'].shape[1])
            # graph = graph.float().to(self.device)
            new_v = new_v + self.implicit_encoder(
                v, self.implicit_graph.repeat(v.shape[0], 1, 1), graph_alpha
            ) # [batch, num_objs, v_dim]

        # Spatial graph
        if self.spatial_encoder:
            graph = batch['graph'].float().to(self.device)
            new_v = new_v + self.spatial_encoder(v, graph, graph_alpha) # [batch, num_objs, v_dim]
        
        v = new_v
        if graph_alpha:
            v, g_att = v
            return g_att
        return v, q, v_att


class CaptionEncoder(BaseEncoder):
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 c_len: int,
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'GRU',
                 neg_slope: float = 0.01,
    ):
        super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type, att_type)
        
        self.v_net = LReLUNet(v_dim, hidden_dim, neg_slope)
        self.c_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)

        # Caption embedding module
        self.caption_embedding = CaptionEmbedding(
            v_dim=hidden_dim,
            q_dim=hidden_dim,
            c_dim=embed_dim,
            hidden_dim=hidden_dim,
            max_len=c_len,
            device=device,
            dropout=dropout
        )

        # Attention layer for image features based on caption embedding
        self.caption_attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=att_fc_dim)

    def forward(self, batch):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
        Output:
            v: [batch, num_objs, v_dim]
            q, c: [batch, hidden_dim]
            v_att: [batch, num_objs, 1]
        """
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        c = batch['c'].to(self.device)
        cap_len = batch['cap_len'].to(self.device)
        
        # Embed words and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # Get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]
        # Get question-attended visual feature vq
        v = v_att * v # [batch, num_objs, v_dim]

        vq = self.v_net(v.sum(1)) # [batch, hidden_dim]
        q = self.q_net(q) # [batch, hidden_dim]

        c = self.embedding(c) # [batch, c_len, embed_dim]
        c, _ = self.caption_embedding(vq, q, c, cap_len) # [batch, hidden_dim]
        c = self.c_net(c) # [batch, hidden_dim]

        return v, (q, c), v_att