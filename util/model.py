import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from util.modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding
from util.attention import ConcatAttention, DotAttention

def set_model(model_type: str):
    """
    Setup the model according to the model type.
    """
    models = {
        # bottom-up VQA
        'base': BottomUpVQAModel,
        # bottom-up VQA with dot attention
        'new': NewBottomUpVQAModel,
        # VQA with Generating Question Relevant Captions
        'q-caption': QuestionRelevantCaptionsVQAModel,
    }
    keys = '\"/\"'.join(models.keys())
    msg = f'model_type can only be \"{keys}\", but get \"{model_type}\".'
    assert model_type in models.keys(), msg
    return models[model_type]


def use_pretrained_embedding(model, vocab_path: str, device: str):
    """
    Replace the embedding layer in the model with the pre-trained embedding layer.

    Input:
        vocab_path: path for loading pre-trained word vectors
        device: device
    """
    model.embedding = PretrainedWordEmbedding(vocab_path=vocab_path, device=device)
    return model


class BottomUpVQAModel(nn.Module):
    """
    This model is based on the winning entry of the 2017 VQA Challenge, following the system described in 
    'Bottom-Up ad Top-Down Attention for Image Captioning and Visual Question Answering' (https://arxiv.org/abs/1707.07998) and 
    'Tips and Tricks for Visual Question Answering: Learning from teh 2017 Challenge' (https://arxiv.org/abs/1708.02711)

    Code reference: https://github.com/hengyuan-hu/bottom-up-attention-vqa
    """

    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 ans_dim: int,
                 cls_layer=2,
                 dropout=0.5
    ):
        """Input:
            # for question embedding
            ntoken: number of tokens (i.e. size of vocabulary)
            embed_dim: dimension of embedding
            hidden_dim: dimension of hidden layers
            rnn_layer: number of RNN layers

            # for attention
            v_dim: dimension of image features
            att_fc_dim: dimension of attention fc layer

            # for classifier
            ans_dim: dimension of output (i.e. number of answer candidates)
            cls_layer: number of non-linear layers in the classifier (default=2)
            dropout: dropout (default=0.5)
        """

        super().__init__()
        # word embedding for question
        self.embedding = nn.Embedding(ntoken+1, embed_dim, padding_idx=ntoken)
        
        # RNN for question
        self.q_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=rnn_layer,
            dropout=dropout
        )
        
        # dropout
        self.dropout = nn.Dropout(dropout)

        # attention layer for image features based on questions
        self.attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=att_fc_dim)

        # non-linar layers for image features
        self.q_net = FCNet(hidden_dim, hidden_dim)

        # non-linear layers for question
        self.v_net = FCNet(v_dim, hidden_dim)

        # classifier
        self.classifier = FCNet(
            in_dim=hidden_dim,
            mid_dim=2*hidden_dim,
            out_dim=ans_dim,
            layer=cls_layer, dropout=dropout
        )

    def forward(self, v, q):
        """
        Input:
            v: [batch, v_len, v_dim]
            q: [batch, q_len]
        Output:[batch, num_answer_candidate]
        """
        
        # embed words and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]
        # sum-up visual features
        v = (v_att * v).sum(1) # [batch, v_dim]
        
        # FC layers
        q = self.q_net(q) # [batch, hidden_dim]
        v = self.v_net(v) # [batch, hidden_dim]
        
        # fuse visual and question features (multiplication here)
        joint = q * v # [batch, hidden_dim]
        
        # predict answer
        joint = self.classifier(joint) # [batch, num_answer_candidate]
        return joint


class NewBottomUpVQAModel(BottomUpVQAModel):
    """
    This model is based on the winning entry of the 2017 VQA Challenge,
    but replaces the concat attention in the original design with the dot attention.
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 ans_dim: int,
                 cls_layer=2,
                 dropout=0.5
    ):
        """Input:
            # for question embedding
            ntoken: number of tokens (i.e. size of vocabulary)
            embed_dim: dimension of question embedding
            hidden_dim: dimension of hidden layers
            rnn_layer: number of RNN layers

            # for attention
            v_dim: dimension of image features
            att_fc_dim: dimension of attention fc layer

            # for classifier
            ans_dim: dimension of output (i.e. number of answer candidates)
            cls_layer: number of non-linear layers in the classifier (default=2)
            dropout: dropout (default=0.5)
        """
        super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, ans_dim, dropout)
        self.attention = DotAttention(v_dim, hidden_dim, att_fc_dim) # replace the attention
        # the forward process is the same


class QuestionRelevantCaptionsVQAModel(BottomUpVQAModel):
    """
    This model follows the system described in 
    'Generating Question Relevant Captions to Aid Visual Question Answering' (https://arxiv.org/abs/1906.00513)
    """

    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 ans_dim: int,
                 cls_layer=2,
                 dropout=0.5
    ):
        """Input:
            # for question embedding
            ntoken: number of tokens (i.e. size of vocabulary)
            embed_dim: dimension of question embedding
            hidden_dim: dimension of hidden layers
            rnn_layer: number of RNN layers

            # for attention
            v_dim: dimension of image features
            att_fc_dim: dimension of attention fc layer

            # for classifier
            ans_dim: dimension of output (i.e. number of answer candidates)
            cls_layer: number of non-linear layers in the classifier (default=2)
            dropout: dropout (default=0.5)
        """
        super().__init__(ntoken, embed_dim, hidden_dim, rnn_layer, v_dim, att_fc_dim, ans_dim, dropout)

        # caption embedding

        # attention layer for image features based on captions
        self.caption_attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=att_fc_dim)

    def forward_cap(self, v, q, c):
        """
        Forward function for caption generation.
        
        Input:
            v: [batch, v_len, v_dim]
            c: [batch, c_len]
            q: [batch, q_len]
        Output:[batch, c_len, vocab_dim]
        """
        return

    def forward(self, v, q, c):
        """
        Forward function for VQA prediction.

        Input:
            v: [batch, v_len, v_dim]
            c: [batch, c_len]
            q: [batch, q_len]
        Output:[batch, num_answer_candidate]
        """
        return