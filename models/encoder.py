import numpy as np

import torch
import torch.nn as nn

from util.modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding, CaptionEmbedding, LReLUNet
from util.attention import ConcatAttention, MultiplyAttention


# This model is based on the winning entry of the 2017 VQA Challenge, following the system described in 
# 'Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering' (https://arxiv.org/abs/1707.07998) and 
# 'Tips and Tricks for Visual Question Answering: Learning from teh 2017 Challenge' (https://arxiv.org/abs/1708.02711)
#
# Code reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa

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
        self.attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=att_fc_dim)

    def forward(self, batch):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
        Output:
            v: [batch, num_objs, v_dim]
            q: [batch, hidden_dim]
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
        return v, q

class NewEncoder(BaseEncoder):
    """
    This is for the winning entry of the 2017 VQA Challenge,
    but replaces the concat attention in the original design with the dot attention.
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
    ):
        super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, device, dropout, rnn_type)
        self.attention = MultiplyAttention(v_dim, hidden_dim, att_fc_dim)

