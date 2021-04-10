import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.weight_norm import weight_norm

from .modules import FCNet
from .attention import set_att


def set_decoder(  decoder_type: str,
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
    )


class DecoderModule(nn.Module):
    def __init__(self): super().__init__()

    def init_hidden(self, batch):
        """Initialize hidden states."""
        init = torch.zeros((batch, self.hidden_dim)).to(self.device)
        if self.rnn_type == 'LSTM': return (init, init)
        else: return init

    def select_hidden(self, h, batch):
        if self.rnn_type == 'LSTM': return (h[0][:batch], h[1][:batch])
        else: return h[:batch]

    def forward(self, v, caption, cap_len): return


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

        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.rnn = rnn_cls(input_size=embed_dim+v_dim, hidden_size=hidden_dim)

        self.embedding = nn.Embedding(ntoken, embed_dim)
        self.attention = set_att(att_type)(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim)
        self.fcnet = nn.Linear(hidden_dim, ntoken)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, v, caption, cap_len):
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
        h = self.init_hidden(batch)

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
            h = self.select_hidden(h, batch_t) # h: [batch_t, hidden_dim]
            
            # Attention of image considering hidden state
            h0 = h[0] if self.rnn_type == 'LSTM' else h
            att = self.attention(v[:batch_t], h0) # [batch, num_objs, 1]
            att_v = (att * v[:batch_t]).sum(1) # [batch, v_dim]
            att = att.squeeze()

            # Decode
            h = self.rnn(torch.cat([caption[:batch_t,t,:], att_v], dim=1), h)
            h0 = h[0] if self.rnn_type == 'LSTM' else h

            # Predict the possible output word
            h0 = self.fcnet(h0) # [batch_t, ntoken]
            
            # Save the results
            output[:batch_t, t, :] = h0
            alphas[:batch_t, t, :] = att
        
        # Softmax
        output = self.softmax(output)
        
        # Since decode starting with <start>, the targets are all words after <start>
        sort_caption = sort_caption[:,1:]
        
        return {
            'predict': pack_padded_sequence(output, decode_len, batch_first=True).data,
            'target': pack_padded_sequence(sort_caption, decode_len, batch_first=True).data,
        }


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

        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.word_rnn = rnn_cls(input_size=hidden_dim + v_dim + embed_dim, hidden_size=hidden_dim)
        self.language_rnn = rnn_cls(input_size=v_dim + hidden_dim, hidden_size=hidden_dim)

        self.embedding = nn.Embedding(ntoken, embed_dim)
        self.attention = set_att(att_type)(v_dim=v_dim, q_dim=hidden_dim, hidden_dim=hidden_dim)
        self.h1_fcnet = nn.Linear(hidden_dim, hidden_dim)
        self.h2_fcnet = nn.Linear(hidden_dim, ntoken)
        self.softmax = nn.Softmax(dim=1)

    def decode(self, v, prev, h1, h2):
        """Decode process
        Input:
            v: visual features[batch, num_objs, v_dim]
            prev: previous decoded results (initial: <start>)
            h1, h2: hidden states [batch, hidden_dim]
        Output:
            h: next word [batch, ntoken]
            h1, h2: hidden states [batch, hidden_dim]
            att: [batch, num_objs]
        """
        # Flatten image features
        v_mean = v.mean(1).to(self.device) # [batch, v_dim]

        # Encode captions
        caption = self.embedding(prev) # [batch, embed_dim]

        # First RNN: Word RNN
        h = h2[0] if self.rnn_type == 'LSTM' else h2
        h1 = self.word_rnn(
            torch.cat([h, v_mean, caption ], dim=1) # x1: [batch, hidden_dim + v_dim + embed_dim]
            , h1                                    # h1: [batch, hidden_dim]
        )                                           # output: [batch, hidden_dim]
        h = h1[0] if self.rnn_type == 'LSTM' else h1
        h = self.h1_fcnet(h)

        # Attention
        att = self.attention(v, h) # [batch, num_objs, 1]
        att_v = (att * v).sum(1) # [batch, v_dim]

        # Second RNN: Language RNN
        h2 = self.language_rnn(
            torch.cat([att_v, h], dim=1)    # x2: [batch, v_dim + hidden_dim]
            , h2                            # h2: [batch, hidden_dim]
        )                                   # output: [batch, hidden_dim]
        h = h2[0] if self.rnn_type == 'LSTM' else h2

        # Predict the possible output word
        h = self.softmax(self.h2_fcnet(h)) # [batch_t, ntoken]
        return h, h1, h2, att.squeeze()

    def forward(self, v, caption, cap_len):
        """Training process
        Input:
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
        total_h1 = self.init_hidden(batch)
        total_h2 = self.init_hidden(batch)

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
            h1 = self.select_hidden(total_h1, batch_t) # h1: [batch_t, hidden_dim]
            h2 = self.select_hidden(total_h2, batch_t) # h2: [batch_t, hidden_dim]

            # First RNN: Word RNN
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

            # Second RNN: Language RNN
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
        return {
            'predict': output[restore_id,:,:],
            'target': sort_caption[:, 1:], # Since decode starting with <start>, the targets are all words after <start>
            'decode_len': decode_len,
            'batches': batches,
            'alpha': alphas[restore_id,:,:],
        }