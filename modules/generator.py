import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.weight_norm import weight_norm

from .modules import FCNet
from .attention import set_att


def set_decoder(decoder_type: str,
                ntoken: int,
                embed_dim: int,
                hidden_dim: int,
                v_dim: int,
                max_len: int,
                device: str,
                dropout: float,
                rnn_type: str,
                att_type: str,
    ):
    if decoder_type == 'none': return
    return {
        'base': BaseDecoder,
        'butd': BUTDDecoder
    }[decoder_type](
        ntoken=ntoken,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        v_dim=v_dim,
        max_len=max_len,
        device=device,
        dropout=dropout,
        rnn_type=rnn_type,
        att_type=att_type
    ).to(device)


class DecoderModule(nn.Module):
    def init(self):
        super().__init__()
        self.h_num = 1
    
    def init_hidden(self, batch_size):
        """Initialize hidden states."""
        init = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        if self.rnn_type == 'LSTM': return [(init, init)] * self.h_num
        else: return [init] * self.h_num

    def select_hidden(self, h, batch_size):
        output = []
        for i in range(len(h)):
            if self.rnn_type == 'LSTM': output.append(h[i][0][:batch_size], h[i][1][:batch_size])
            else: output.append(h[i][:batch_size])
        return output
        # if self.rnn_type == 'LSTM': return (h[0][:batch_size], h[1][:batch_size])
        # else: return h[:batch_size]

    def decode(self, v, v_mean, prev, h):
        assert False, 'Error: please override the decode function'

    def forward(self, batch):
        """Training process
        """
        v = batch['v'].to(self.device)
        caption = batch['c'].to(self.device)
        cap_len = batch['cap_len'].to(self.device)
        target = batch['c_target'].to(self.device)
        num_objs = v.size(1)
        
        # Flatten image features
        v_mean = v.mean(1) # [batch, v_dim]

        # Sort input data by decreasing lengths, so that we can process only valid time steps, i.e., no need to process the <pad>
        cap_len, sort_id = cap_len.sort(dim=0, descending=True)
        restore_id = sorted(sort_id, key=lambda k: sort_id[k]) # to restore the order
        caption = caption[sort_id]
        v = v[sort_id]
        v_mean = v_mean[sort_id]

        # Initialize RNN states
        batch_size = caption.size(0)
        h = self.init_hidden(batch_size)

        # Create tensor to hold the caption embedding after all time steps
        output = torch.zeros(batch_size, self.max_len, self.ntoken, device=self.device)
        # alphas = torch.zeros(batch, self.max_len, num_objs).to(self.device)

        # We don't decode at the <end> position
        decode_len = (cap_len - 1).tolist()

        # This list if for saving the batch size for each time step
        batches = []
        # For each time step:
        for t in range(max(decode_len)):
            # Only generate captions which is longer than t (ignore <pad>)
            batch_t = sum([l > t for l in decode_len])
            batches.append(batch_t)
            h = self.select_hidden(h, batch_t) # h: [batch_t, hidden_dim]
            
            # Save the results
            h, word = self.decode(
                v=v[:batch_t],
                v_mean=v_mean[:batch_t],
                prev=caption[:batch_t, t, :],
                h=h
            )
            output[:batch_t, t, :] = word
        
        # Since decode starting with <start>, the targets are all words after <start>
        target = target[:,1:]
        
        return {
            'predict': pack_padded_sequence(output, decode_len, batch_first=True).data,
            'target': pack_padded_sequence(target, decode_len, batch_first=True).data,
        }


class BaseDecoder(DecoderModule):
    """
    Base generator based on "Show, Attend and Tell"
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 v_dim: int,
                 max_len: int,
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'GRU',
                 att_type: str = 'base',
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
                rnn_type: choose the type of RNN (default = GRU)
        """
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.ntoken = ntoken
        self.device = device
        self.h_num = 1

        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.rnn = rnn_cls(input_size=embed_dim+v_dim, hidden_size=hidden_dim)

        self.attention = set_att(att_type)(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim)
        self.fcnet = nn.Linear(hidden_dim, ntoken)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.fcnet.bias.data.fill_(0)
        self.fcnet.weight.data.uniform_(-0.1, 0.1)

    def decode(self, v, v_mean, prev, h):
        """Decode process
        """
        # Attention of image considering hidden state
        h = h[0]
        h0 = h[0] if self.rnn_type == 'LSTM' else h
        att = self.attention(v, h0) # [batch, num_objs, 1]
        att_v = (att * v).sum(1) # [batch, v_dim]

        # Decode
        h = self.rnn(torch.cat([prev, att_v], dim=1), h)
        h0 = h[0] if self.rnn_type == 'LSTM' else h
        return [h0], self.fcnet(self.dropout(h0))


class BUTDDecoder(DecoderModule):
    """
    Caption decoder mentioned in 'Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering'
    """
    def __init__(self,
                 ntoken: int,
                 embed_dim: int,
                 hidden_dim: int,
                 v_dim: int,
                 max_len: int,
                 device: str,
                 dropout: float = 0.5,
                 rnn_type: str = 'GRU',
                 att_type: str = 'base',
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
                rnn_type: choose the type of RNN (default = GRU)
        """
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.ntoken = ntoken
        self.device = device
        self.h_num = 2

        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.word_rnn = rnn_cls(input_size=hidden_dim + v_dim + embed_dim, hidden_size=hidden_dim)
        self.language_rnn = rnn_cls(input_size=v_dim + hidden_dim, hidden_size=hidden_dim)

        self.attention = set_att(att_type)(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim)
        self.h1_fcnet = nn.Linear(hidden_dim, hidden_dim)
        self.h2_fcnet = nn.Linear(hidden_dim, ntoken)
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        self.h1_fcnet.bias.data.fill_(0)
        self.h1_fcnet.weight.data.uniform_(-0.1, 0.1)
        self.h2_fcnet.bias.data.fill_(0)
        self.h2_fcnet.weight.data.uniform_(-0.1, 0.1)

    def decode(self, v, v_mean, prev, h):
        h1, h2 = h

        # First RNN: Word RNN
        h = h2[0] if self.rnn_type == 'LSTM' else h2
        h1 = self.word_rnn(torch.cat([h, v_mean, prev], dim=1), h1) # output: [batch_t, hidden_dim]
        h = h1[0] if self.rnn_type == 'LSTM' else h1
        h = self.h1_fcnet(self.dropout(h))

        # Attention
        att = self.attention(v, h) # [batch_t, num_objs, 1]
        att_v = (att * v).sum(1) # [batch_t, v_dim]
        self.att = att # save the attention score

        # Second RNN: Language RNN
        h2 = self.language_rnn(torch.cat([att_v, h], dim=1), h2 )# output: [batch_t, hidden_dim]
        h = h2[0] if self.rnn_type == 'LSTM' else h2
        return [h1, h2], self.h2_fcnet(self.dropout(h))