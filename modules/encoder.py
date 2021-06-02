import numpy as np

import torch
import torch.nn as nn

from .gcn import GCN
from .modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding, CaptionEmbedding, LReLUNet
from .attention import set_att

def set_encoder(encoder_type: str,
                ntoken: int,
                v_dim: int,
                embed_dim: int,
                hidden_dim: int,
                device: str,
                dropout: float,
                rnn_type: str,
                rnn_layer: int,
                att_type: str,
                conv_type: str,
                conv_layer: int,
                vocab_path: str = '',
    ):
    if encoder_type == 'base':
        model = BaseEncoder(
            ntoken=ntoken,
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=device,
            dropout=dropout,
            rnn_type=rnn_type,
            rnn_layer=rnn_layer,
            att_type=att_type
        )
    if encoder_type == 'relation':
        model = RelationEncoder(
            ntoken=ntoken,
            v_dim=v_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            device=device,
            dropout=dropout,
            rnn_type=rnn_type,
            rnn_layer=rnn_layer,
            att_type=att_type,
            conv_type=conv_type,
            conv_layer=conv_layer
        )
    if encoder_type == 'cap':
        model = CaptionEncoder(
            ntoken=ntoken,
            embed_dim=embed_dim,
            device=device,
        )
    if vocab_path != '':
        model.embedding = PretrainedWordEmbedding(vocab_path=vocab_path, device=device)
    return model.to(device)

# This model is based on the winning entry of the 2017 VQA Challenge, following the system described in 
# 'Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering' (https://arxiv.org/abs/1707.07998) and 
# 'Tips and Tricks for Visual Question Answering: Learning from teh 2017 Challenge' (https://arxiv.org/abs/1708.02711)
#
# Code reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa

class CaptionEncoder(nn.Module):
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 device: str,
    ):
        super().__init__()
        self.device = device

        # Word embedding for question
        self.embedding = nn.Embedding(ntoken+1, embed_dim, padding_idx=ntoken)
    
    def forward(self, batch):
        # Caption embedding
        c_target = batch['c'].to(self.device)
        c = self.embedding(c_target) # [batch, c_len, embed_dim]
        
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        c_target = batch['c'].to(self.device)
        cap_len = batch['cap_len'].to(self.device)

        return {
            'v': batch['img'],          # [batch, num_objs, v_dim]
            'c': c,                     # [batch, c_len, embed_dim]
            'c_target': c_target,       # [batch, c_len]
            'cap_len': batch['cap_len'],# [batch]
        }

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
        self.attention = set_att(att_type)(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim)

        # Non-linear layers for image features
        self.q_net = FCNet(hidden_dim, hidden_dim)

    def base_forward(self, batch):
        """
        Input:
            v: [batch, num_objs, v_dim]
            q: [batch, q_len]
        """
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        c_target = batch['c'].to(self.device)
        cap_len = batch['cap_len'].to(self.device)
        
        # Embed words in question and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # Get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]

        # Get question-attended visual feature vq
        v = v_att * v # [batch, num_objs, v_dim]

        # Question embedding
        q = self.q_net(q) # [batch, hidden_dim]

        # Caption embedding
        c = self.embedding(c_target) # [batch, c_len, embed_dim]

        return {
            'v': v,                 # [batch, num_objs, v_dim]
            'q': q,                 # [batch, hidden_dim]
            'c': c,                 # [batch, c_len, embed_dim]
            'c_target': c_target,   # [batch, c_len]
            'cap_len': cap_len,     # [batch]
            'v_att': v_att          # [batch, num_objs, 1]
        }

    def forward(self, batch): return self.base_forward(batch)


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
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'GRU',
                 att_type: str = 'base',
                 conv_layer: int = 1,
                 conv_type: str = 'corr',
                 use_imp: bool = False,
                 use_spa: bool = True,
                 use_sem: bool = False,
                 num_objs: int = 36
    ):
        super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, device, dropout, rnn_type, att_type)
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
        output = self.base_forward(batch)

        # Get relation-aware visual feature
        output_v = torch.zeros_like(output['v'])
        g_att = []
        batch_size = output['v'].size(0)
        # Implicit graph
        if self.implicit_encoder:
            # graph = torch.ones_like(batch['graph']) - torch.eye(batch['graph'].shape[1])
            # graph = graph.float().to(self.device)
            new_v = self.implicit_encoder(
                output['v'], self.implicit_graph.repeat(batch_size, 1, 1), graph_alpha
            ) # [batch, num_objs, v_dim]
            if graph_alpha: new_v, g_att = new_v
            output_v += new_v

        # Spatial graph
        if self.spatial_encoder:
            graph = batch['graph'].float().to(self.device)
            new_v = self.spatial_encoder(output['v'], graph, graph_alpha) # [batch, num_objs, v_dim]
            if graph_alpha: new_v, g_att = new_v
            output_v += new_v
        
        if graph_alpha: return g_att
        output['v'] = output_v
        return output


# class CaptionEncoder(BaseEncoder):
#     def __init__(self,
#                  ntoken: int,
#                  embed_dim: int,
#                  hidden_dim: int,
#                  rnn_layer: int,
#                  v_dim: int,
#                  c_len: int,
#                  device: str,
#                  dropout: float = 0.5,
#                  rnn_type: str = 'GRU',
#                  att_type: str = 'base',
#                  neg_slope: float = 0.01,
#     ):
#         super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, device, dropout, rnn_type, att_type)
        
#         self.v_net = LReLUNet(v_dim, hidden_dim, neg_slope)
#         self.c_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)

#         # Caption embedding module
#         self.caption_embedding = CaptionEmbedding(
#             v_dim=hidden_dim,
#             q_dim=hidden_dim,
#             c_dim=embed_dim,
#             hidden_dim=hidden_dim,
#             max_len=c_len,
#             device=device,
#             dropout=dropout
#         )

#         # Attention layer for image features based on caption embedding
#         self.caption_attention = set_att('base')(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim)

#     def forward(self, batch):
#         """
#         Input:
#             v: [batch, num_objs, v_dim]
#             q: [batch, q_len]
#         Output:
#             v: [batch, num_objs, v_dim]
#             q, c: [batch, hidden_dim]
#             v_att: [batch, num_objs, 1]
#         """
#         # Setup inputs
#         v = batch['img'].to(self.device)
#         q = batch['q'].to(self.device)
#         c = batch['c'].to(self.device)
#         cap_len = batch['cap_len'].to(self.device)
        
#         # Embed words and take the last output of RNN layer as the question embedding
#         q = self.embedding(q) # [batch, q_len, q_embed_dim]
#         q = self.q_rnn(q) # [batch, hidden_dim]
        
#         # Get the attention of visual features based on question embedding
#         v_att = self.attention(v, q) # [batch, num_objs, 1]
#         # Get question-attended visual feature vq
#         v = v_att * v # [batch, num_objs, v_dim]

#         vq = self.v_net(v.sum(1)) # [batch, hidden_dim]
#         q = self.q_net(q) # [batch, hidden_dim]

#         c = self.embedding(c) # [batch, c_len, embed_dim]
#         c, _ = self.caption_embedding(vq, q, c, cap_len) # [batch, hidden_dim]
#         c = self.c_net(c) # [batch, hidden_dim]

#         return v, (q, c), v_att