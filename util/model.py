import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from util.modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding, CaptionEmbedding, LReLUNet
from util.attention import ConcatAttention, MultiplyAttention

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
                 device: str,
                 cls_layer=2,
                 dropout=0.5
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
            For classifier:
                ans_dim: dimension of output (i.e. number of answer candidates)
                cls_layer: number of non-linear layers in the classifier (default=2)
            Others:
                device: device
                dropout: dropout (default = 0.5)
        """

        super().__init__()
        # Word embedding for question
        self.embedding = nn.Embedding(ntoken+1, embed_dim, padding_idx=ntoken)
        
        # RNN for question
        self.q_rnn = SentenceEmbedding(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            rnn_layer=rnn_layer,
            dropout=dropout,
            device=device
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Attention layer for image features based on questions
        self.attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=att_fc_dim)

        # Non-linar layers for image features
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

    def input_embedding(self, v, q):
        # Embed words and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # Get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]
        # Get question-attended visual feature vq
        v = v_att * v # [batch, num_objs, v_dim]
        return v, q

    def forward(self, v, q):
        """
        Input:
            v: [batch, v_len, v_dim]
            q: [batch, q_len]
        Output:[batch, num_answer_candidate]
        """
        
        ##########################################################################################
        # # Embed words and take the last output of RNN layer as the question embedding
        # q = self.embedding(q) # [batch, q_len, q_embed_dim]
        # q = self.q_rnn(q) # [batch, hidden_dim]
        
        # # Get the attention of visual features based on question embedding
        # v_att = self.attention(v, q) # [batch, num_objs, 1]
        # # Get question-attended visual feature vq
        # v = (v_att * v).sum(1) # [batch, v_dim]
        ##########################################################################################
        v, q = self.input_embedding(v, q)
        v = v.sum(1) # [batch, v_dim]
        
        # FC layers
        q = self.q_net(q) # [batch, hidden_dim]
        v = self.v_net(v) # [batch, hidden_dim]
        
        # Fuse visual and question features (multiplication here)
        joint = q * v # [batch, hidden_dim]
        
        # Predict answer
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
                 device: str,
                 cls_layer=2,
                 dropout=0.5
    ):
        # Replace the attention module
        # The forward process is the same
        super().__init__(
            ntoken=ntoken, embed_dim=embed_dim, hidden_dim=hidden_dim, rnn_layer=rnn_layer,
            v_dim=v_dim, att_fc_dim=att_fc_dim, ans_dim=ans_dim,
            device=device, cls_layer=cls_layer, dropout=dropout
        )
        self.attention = MultiplyAttention(v_dim, hidden_dim, att_fc_dim)
        

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
                 c_len: int,
                 v_dim: int,
                 att_fc_dim: int,
                 ans_dim: int,
                 device: str,
                 cls_layer: int = 2,
                 dropout: float = 0.5,
                 neg_slope: float = 0.01
    ):
        """Input:
            For question embedding:
                ntoken: number of tokens (i.e. size of vocabulary)
                embed_dim: dimension of question embedding
                hidden_dim: dimension of hidden layers
                rnn_layer: number of RNN layers
            For caption embedding
                c_len: the maximal length of captions
            For attention:
                v_dim: dimension of image features
                att_fc_dim: dimension of attention fc layer
            For classifier:
                ans_dim: dimension of output (i.e. number of answer candidates)
                cls_layer: number of non-linear layers in the classifier (default=2)
            Others:
                device: device
                dropout: dropout (default = 0.5)
                neg_slope: negative slope for Leaky ReLU (default = 0.01)
        """

        ##########################################################################################
        # Image and Question Embedding
        ##########################################################################################
        super().__init__(
            ntoken=ntoken, embed_dim=embed_dim, hidden_dim=hidden_dim, rnn_layer=rnn_layer,
            v_dim=v_dim, att_fc_dim=att_fc_dim, ans_dim=ans_dim,
            device=device, cls_layer=cls_layer, dropout=dropout
        )

        ##########################################################################################
        # Caption Embedding
        ##########################################################################################
        # Caption embedding module
        self.caption_embedding = CaptionEmbedding(
            v_dim=v_dim,
            q_dim=embed_dim,
            hidden_dim=hidden_dim,
            max_len=c_len,
            device=device,
            dropout=dropout
        )

        # Attention layer for image features based on caption embedding
        self.caption_attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=att_fc_dim)

        ##########################################################################################
        # VQA Module
        ##########################################################################################
        # For caption-attended visual features
        self.c_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.vq_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.joint_c_vq = LReLUNet(hidden_dim, hidden_dim, neg_slope)

        # For incorporating the information from the captions
        self.vqc_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.final_net = nn.Sequential([
            LReLUNet(hidden_dim, ans_dim, neg_slope),
            nn.Sigmoid()
        ])

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

    def forward(self, v, q, c, cap_len):
        """
        Forward function for VQA prediction.

        Input:
            v: visual features [batch, v_len, v_dim]
            q: question tokens [batch, q_len]
            c: caption tokens [batch, c_len (this is the maximal length of captions, not equal to cap_len)]
            cap_len: ground truth caption length [batch, 1]
        Output:[batch, num_answer_candidate]
        """

        ##########################################################################################
        # Image and Question Embedding
        ##########################################################################################
        # Embed question tokens and take the last output of RNN layer as the question embedding
        v, q = self.input_embedding(v, q)
        vq = v.sum(1) # [batch, v_dim]

        ##########################################################################################
        # Caption Embedding
        ##########################################################################################
        # Embed caption tokens and compute the caption embedding
        c = self.embedding(c) # [batch, c_len, embed_dim]
        vq = self.v_net(vq) # [batch, hidden_dim]
        q = self.q_net(q) # [batch, hidden_dim]
        c, _ = self.caption_embedding(vq, q, c, cap_len) # [batch, hidden_dim]

        ##########################################################################################
        # VQA Module
        ##########################################################################################
        # Produce caption-attended features
        v = self.vq_net(v) # [batch, num_objs, hidden_dim]
        c = self.c_net(c) # [batch, hidden_dim]
        joint = self.joint_c_vq(c * v)
        joint = nn.functional.softmax(joint, 1) # [batch, num_objs, hidden_dim]
        v = (joint * v).sum(1) # [batch, hidden_dim]

        # To better incorporate the information from the captions into the VQA process, add the caption feature ot the attended image features,
        # and then element-wise multiply by the question features.
        v = self.vqc_net(v) # [batch, hidden_dim]
        joint = q * (v + c) # [batch, hidden_dim]
        joint = self.final_net(joint) # [batch, ans_dim]

        return joint