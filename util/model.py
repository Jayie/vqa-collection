import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.weight_norm import weight_norm

from util.modules import FCNet, SentenceEmbedding, PretrainedWordEmbedding, CaptionEmbedding, LReLUNet
from util.attention import ConcatAttention, MultiplyAttention

def set_model(model_type: str):
    """Setup the model according to the model type."""
    models = {
        # bottom-up VQA
        'base': BottomUpVQAModel,
        # bottom-up VQA with dot attention
        'new': NewBottomUpVQAModel,
        # VQA-E
        'vqa-e': VQAEModel,
        # VQA with Generating Question Relevant Captions
        'q-caption': QuestionRelevantCaptionsVQAModel,
    }
    keys = '\"/\"'.join(models.keys())
    msg = f'model_type can only be \"{keys}\", but get \"{model_type}\".'
    assert model_type in models.keys(), msg
    return models[model_type]


def use_pretrained_embedding(model, vocab_path: str, device: str):
    """ Replace the embedding layer in the model with the pre-trained embedding layer.
    Input:
        vocab_path: path for loading pre-trained word vectors
        device: device
    """
    model.embedding = PretrainedWordEmbedding(vocab_path=vocab_path, device=device)
    return model


def instance_bce_with_logits(predict, target):
    """ Loss function for VQA prediction"""
    assert predict.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(predict, target)
    loss *= target.size(1)
    return loss


def ce_for_language_model(predict, target):
    """ Loss function for caption generation"""
    assert predict.dim() == 2
    loss = nn.functional.cross_entropy(predict, target)
    return loss

def ce_for_relevant_captions(vqa_loss, cap_loss, captions, v):
    # TODO: loss_c_i for 'Generating Question Relevant Captions to Aid Visual Question Answering' (https://arxiv.org/abs/1906.00513)
    vqa_loss.backward()
    vqa_grad = v.grad().sum(dim=1) # SUM(d(vqa logit) / d(vqk)), k in range(36) [batch, v_dim]
    cap_loss.backward()
    

    
    return

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
                 cls_layer: int = 2,
                 dropout: float = 0.5
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
        self.device = device
        self.return_loss = True

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

    def set_return_loss(self, return_loss=True):
        self.return_loss = return_loss

    def input_embedding(self, v, q):
        # Embed words and take the last output of RNN layer as the question embedding
        q = self.embedding(q) # [batch, q_len, q_embed_dim]
        q = self.q_rnn(q) # [batch, hidden_dim]
        
        # Get the attention of visual features based on question embedding
        v_att = self.attention(v, q) # [batch, num_objs, 1]
        # Get question-attended visual feature vq
        v = v_att * v # [batch, num_objs, v_dim]
        return v, q

    def forward_vqa(self, v, q):
        visual_feature, q = self.input_embedding(v, q)
        v = visual_feature.sum(1) # [batch, v_dim]
        
        # FC layers
        q = self.q_net(q) # [batch, hidden_dim]
        v = self.v_net(v) # [batch, hidden_dim]
        
        # Fuse visual and question features (multiplication here)
        joint = q * v # [batch, hidden_dim]
        
        # Predict answer
        joint = self.classifier(joint) # [batch, num_answer_candidate]
        
        # Return: joint = logits of predictor, v = visual embeddings (for caption generator)
        return joint, visual_feature

    def forward(self, batch):
        """
        Input:
            v: [batch, v_len, v_dim]
            q: [batch, q_len]
        Output:[batch, num_answer_candidate]
        """
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        target = batch['a'].float().to(self.device)
        predict, v = self.forward_vqa(v, q)

        if self.return_loss:
            loss = instance_bce_with_logits(predict, target)
            return predict, loss
        return predict



class NewBottomUpVQAModel(BottomUpVQAModel):
    """
    This model is based on the winning entry of the 2017 VQA Challenge, but replaces the concat attention in the original design with the dot attention.
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
        


class VQAEModel(NewBottomUpVQAModel):
    """
    This model follows the system described in 'VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions' (https://arxiv.org/abs/1803.07464)
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 c_len: int,
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
            For attention:
                v_dim: dimension of image features
                att_fc_dim: dimension of attention fc layer
            For caption embedding
                c_len: the maximal length of captions
            For classifier:
                ans_dim: dimension of output (i.e. number of answer candidates)
                cls_layer: number of non-linear layers in the classifier (default=2)
            Others:
                device: device
                dropout: dropout (default = 0.5)
                neg_slope: negative slope for Leaky ReLU (default = 0.01)
        """

        # VQA module
        super().__init__(
            ntoken=ntoken, embed_dim=embed_dim, hidden_dim=hidden_dim, rnn_layer=rnn_layer,
            v_dim=v_dim, att_fc_dim=att_fc_dim, ans_dim=ans_dim,
            device=device, cls_layer=cls_layer, dropout=dropout
        )

        # Caption generator
        self.generator = CaptionDecoder(
            ntoken=ntoken,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            v_dim=v_dim,
            max_len=c_len,
            device=device,
            dropout=dropout,
        )

    def forward_cap(self, v, c, cap_len):
        predict, target, decode_len, _, _, _ = self.generator(v, c, cap_len)
        # Remove time steps that we didn't decode at (i.e. are <pad>)
        predict = pack_padded_sequence(predict, decode_len, batch_first=True).data # [num_non_pad_tokens, vocab_dim]
        target = pack_padded_sequence(target, decode_len, batch_first=True).data # [num_non_pad_tokens, 1]
        return predict, target

    def forward(self, batch):
        """
        Input:
            v: [batch, v_len, v_dim]
            q: [batch, q_len]
        Output:[batch, num_answer_candidate]
        """
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        target = batch['a'].float().to(self.device)
        c = batch['c'].to(self.device)
        cap_len = batch['cap_len'].to(self.device)
        
        vqa_predict, v = self.forward_vqa(v, q)
        cap_predict, c = self.forward_cap(v, c, cap_len)
        if self.return_loss:
            loss = instance_bce_with_logits(vqa_predict, target) # VQA loss
            loss += ce_for_language_model(cap_predict, c) # Caption loss
            return vqa_predict, cap_predict, loss
        return vqa_predict, cap_predict



class QuestionRelevantCaptionsVQAModel(BottomUpVQAModel):
    """
    This model follows the system described in 'Generating Question Relevant Captions to Aid Visual Question Answering' (https://arxiv.org/abs/1906.00513)
    """

    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 rnn_layer: int,
                 v_dim: int,
                 att_fc_dim: int,
                 c_len: int,
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
            For attention:
                v_dim: dimension of image features
                att_fc_dim: dimension of attention fc layer
            For caption embedding
                c_len: the maximal length of captions
            For classifier:
                ans_dim: dimension of output (i.e. number of answer candidates)
                cls_layer: number of non-linear layers in the classifier (default=2)
            Others:
                device: device
                dropout: dropout (default = 0.5)
                neg_slope: negative slope for Leaky ReLU (default = 0.01)
        """

        ##########################################################################################
        # VQA Module
        ##########################################################################################
        # Image and question embedding
        super().__init__(
            ntoken=ntoken, embed_dim=embed_dim, hidden_dim=hidden_dim, rnn_layer=rnn_layer,
            v_dim=v_dim, att_fc_dim=att_fc_dim, ans_dim=ans_dim,
            device=device, cls_layer=cls_layer, dropout=dropout
        )

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

        # VQA cls layer
        # For caption-attended visual features
        self.c_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.vq_net = LReLUNet(v_dim, hidden_dim, neg_slope)
        self.joint_c_vq = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        # For incorporating the information from the captions
        self.vqc_net = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        self.cls_layer = nn.Sequential(
            LReLUNet(hidden_dim, ans_dim, neg_slope),
            nn.Sigmoid()
        )

        ##########################################################################################
        # Caption Generator
        ##########################################################################################
        self.generator = CaptionDecoder(
            ntoken=ntoken,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            v_dim=v_dim,
            max_len=c_len,
            device=device,
            dropout=dropout,
        )
    
    def forward_cap(self, v, c, cap_len):
        predict, target, decode_len, _, _, _ = self.generator(v, c, cap_len)
        # Remove time steps that we didn't decode at (i.e. are <pad>)
        predict = pack_padded_sequence(predict, decode_len, batch_first=True).data # [num_non_pad_tokens, vocab_dim]
        target = pack_padded_sequence(target, decode_len, batch_first=True).data # [num_non_pad_tokens, 1]
        return predict, target

    def forward_vqa(self, v, q, c, cap_len):
        """
        Forward function for VQA prediction.

        Input:
            v: visual features [batch, v_len, v_dim]
            q: question tokens [batch, q_len]
            c: caption tokens [batch, c_len (this is the maximal length of captions, not equal to cap_len)]
            cap_len: ground truth caption length [batch, 1]
        Output:[batch, num_answer_candidate]
        """
        # Image and question embedding
        # Embed question tokens and take the last output of RNN layer as the question embedding
        visual_feature, q = self.input_embedding(v, q)
        vq = visual_feature.sum(1) # [batch, hidden_dim]

        # Caption embedding
        # Embed caption tokens and compute the caption embedding
        c = self.embedding(c) # [batch, c_len, embed_dim]
        vq = self.v_net(vq) # [batch, hidden_dim]
        c, _ = self.caption_embedding(vq, self.q_net(q), c, cap_len) # [batch, hidden_dim]

        # VQA cls layer
        # Produce caption-attended features
        vq = self.vq_net(visual_feature) # [batch, num_objs, hidden_dim]
        c = self.c_net(c) # [batch, hidden_dim]
        joint = self.joint_c_vq(c.unsqueeze(1).repeat(1, vq.size(1), 1) * vq)
        joint = nn.functional.softmax(joint, 1) # [batch, num_objs, hidden_dim]
        vq = (joint * vq).sum(1) # [batch, hidden_dim]
        # To better incorporate the information from the captions into the VQA process, add the caption feature ot the attended image features,
        # and then element-wise multiply by the question features.
        vq = self.vqc_net(vq) # [batch, hidden_dim]
        joint = q * (vq + c) # [batch, hidden_dim]
        joint = self.cls_layer(joint) # [batch, ans_dim]
        
        # Return: joint (logits of predictor), visual_feature (for caption generator)
        return joint, visual_feature

    def forward(self, batch):
        # Setup inputs
        v = batch['img'].to(self.device)
        q = batch['q'].to(self.device)
        target = batch['a'].float().to(self.device)
        c = batch['c'].to(self.device)
        cap_len = batch['cap_len'].to(self.device)
        

        vqa_predict, v = self.forward_vqa(v, q, c, cap_len)
        cap_predict, c = self.forward_cap(v, c, cap_len)
        if self.return_loss:
            loss = instance_bce_with_logits(vqa_predict, target)
            # TODO:
            # loss += caption loss
            return vqa_predict, cap_predict, loss
        return vqa_predict, cap_predict



class CaptionDecoder(nn.Module):
    """
    Caption generator mentioned in 'Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering'
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 v_dim: int,
                 max_len: int,
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'LSTM'
    ):
        """Input:
            For question embedding:
                ntoken: number of tokens (i.e. size of vocabulary)
                embed_dim: dimension of caption embedding
                hidden_dim: dimension of hidden layers
            For attention:
                v_dim: dimension of image features
                att_fc_dim: dimension of attention fc layer
            For output:
                max_len: the maximal length of captions
            Others:
                device: device
                dropout: dropout (default = 0.5)
                rnn_type: choose the type of RNN (default = LSTM)
        """
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.ntoken = ntoken
        self.device = device

        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.word_rnn = rnn_cls(input_size=hidden_dim + v_dim + embed_dim, hidden_size=hidden_dim)
        self.language_rnn = rnn_cls(input_size=v_dim + hidden_dim, hidden_size=hidden_dim)

        self.embedding = nn.Embedding(ntoken, embed_dim)
        self.attention = ConcatAttention(v_dim=v_dim, q_dim=hidden_dim, fc_dim=hidden_dim)
        self.h1_fcnet = FCNet(in_dim=hidden_dim, out_dim=hidden_dim)
        self.h2_fcnet = FCNet(in_dim=hidden_dim, out_dim=ntoken)
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, shape):
        """Initialize hidden states."""
        init = torch.zeros(shape).to(self.device)
        if self.rnn_type == 'LSTM':
            return (init, init)
        else:
            return init

    def select_hidden(self, h, batch):
        if self.rnn_type == 'LSTM':
            return (h[0][:batch], h[1][:batch])
        else:
            return h[:batch]

    def forward(self, v, caption, cap_len):
        """Input:
            v: visual features [batch, num_objs, v_dim]
            caption: ground truth captions [batch, max_len]
            cap_len: caption lengths for each batch [batch, 1]
        Output:
            predict (Tensor): the decoded captions [batch, max_len, vocab_dim]
            target (Tensor): the sorted ground truth caption tokens [batch, max_len, 1]
            batches (list): the lenght of each batch [batch]
            restore_id (list): the batch IDs to restore the order [batch]
            decode_len (list): the lengths of decoded captions [batch]
            alphas (Tensor): the attention map [batch, num_objs]
        """
        # Flatten image features
        v_mean = v.mean(1).to(self.device) # [batch, v_dim]
        num_objs = v.size(1)

        # Sort input data by decreasing lengths, so that we can process only valid time steps, i.e., no need to process the <pad>
        cap_len, sort_id = cap_len.sort(dim=0, descending=True)
        restore_id = sorted(sort_id, key=lambda k: sort_id[k]) # to restore the order
        sort_caption = caption[sort_id]
        v = v[sort_id]
        v_mean = v_mean[sort_id]

        # Encode captions
        caption = self.embedding(sort_caption) # [batch, max_len, embed_dim]

        # Initialize RNN states
        batch = caption.size(0)
        h1 = self.init_hidden((batch, self.hidden_dim))
        h2 = self.init_hidden((batch, self.hidden_dim))

        # Create tensor to hold the caption embedding after all time steps
        output = torch.zeros(batch, self.max_len, self.ntoken).to(self.device)
        alphas = torch.zeros(batch, self.max_len, num_objs).to(self.device)

        # We don't decode at the <end> position
        decode_len = (cap_len - 1).tolist()

        # This list if for saving the batch size for each time step
        batches = []
        # For each time step:
        for t in range(max(decode_len)):
            # Only generate captions which is longer than t (ignore <pad>)
            batch_t = sum([l > t for l in decode_len])
            batches.append(batch_t)
            h1 = self.select_hidden(h1, batch_t) # h1: [batch_t, hidden_dim]
            h2 = self.select_hidden(h2, batch_t) # h2: [batch_t, hidden_dim]

            # 1: Word RNN
            h = h2[0] if self.rnn_type == 'LSTM' else h2
            h1 = self.word_rnn(
                torch.cat([
                    h,                      # h2: [batch_t, hidden_dim]
                    v_mean[:batch_t],       # v_mean: [batch_t, v_dim]
                    caption[:batch_t, t, :] # caption: [batch_t, embed_dim]
                ], dim=1)                   # x1: [batch_t, hidden_dim + v_dim + embed_dim]
                , h1                        # h1: [batch_t, hidden_dim]
            )                               # output: [batch_t, hidden_dim]
            h = h1[0] if self.rnn_type == 'LSTM' else h1
            h = self.h1_fcnet(h)

            # Attention
            att = self.attention(v[:batch_t], h) # [batch_t, num_objs, 1]
            att_v = (att * v[:batch_t]).sum(1) # [batch_t, v_dim]

            # 2: Language RNN
            h2 = self.language_rnn(
                torch.cat([att_v, h], dim=1)    # x2: [batch_t, v_dim + hidden_dim]
                , h2                            # h2: [batch_t, hidden_dim]
            )                                   # output: [batch_t, hidden_dim]
            h = h2[0] if self.rnn_type == 'LSTM' else h2

            # Predict the possible output word
            h = self.h2_fcnet(h) # [batch_t, ntoken]
            
            # Save the results
            output[:batch_t, t, :] = h
            alphas[:batch_t, t, :] = att.squeeze()
        
        # Softmax
        output = self.softmax(output)

        return output[restore_id,:,:], sort_caption, decode_len, batches, restore_id, alphas[restore_id,:,:]