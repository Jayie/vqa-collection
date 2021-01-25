import numpy as np
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable

from util.attention import CaptionAttention

class FCNet(nn.Module):
    """
    Non-linear fully-connected network.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 mid_dim: int = 0,
                 layer: int = 1,
                 dropout: float = 0
        ):
        """Input:
            in_dim: input dimension
            out_dim: output dimension
            mid_dim: dimension of the layers in the middle of the network (default=0)
            layer: number of layers (default=1)
            dropout: dropout
        """

        super().__init__()
        layers = []

        if layer == 1 or mid_dim == 0:
            # If 1-layer:
            # Layer 1: in_dim -> out_dim
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        else:
            # Else:
            # Suppose there are N layers
            # Layer 1: in_dim -> mid_dim
            layers.append(weight_norm(nn.Linear(in_dim, mid_dim), dim=None))
            layers.append(nn.ReLU())
            
            if dropout != 0:
                    layers.append(nn.Dropout(dropout, inplace=True))
            
            # Layer 2 ~ N-1: mid_dim -> mid_dim
            for _ in range(layer-2):
                layers.append(weight_norm(nn.Linear(mid_dim, mid_dim), dim=None))
                layers.append(nn.ReLU())
                if dropout != 0:
                    layers.append(nn.Dropout(dropout, inplace=True))
            
            # Layer n: mid_dim -> out_dim
            layers.append(weight_norm(nn.Linear(mid_dim, out_dim), dim=None))
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SentenceEmbedding(nn.Module):
    """
    Sentence embedding module.
    """
    def __init__(   self,
                    in_dim: int,
                    hidden_dim: int,
                    rnn_layer: int = 1,
                    dropout: float = 0.5,
                    rnn_type: str = 'LSTM',
                    bidirect: bool = False,
        ):
        """Input:
            in_dim: input dimension (i.e. dimension of word embedding)
            hidden_dim: dimension of the hidden state
            rnn_layer: number of RNN layers
            dropout: dropout
            rnn_type: choose the type of RNN (default=LSTM)
            bidirect: if True, use a bidirectional RNN (default=False)
        """
        super().__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type =='LSTM' else nn.GRU
        
        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size = hidden_dim,
            num_layers = rnn_layer,
            dropout=dropout,
            bidirectional=bidirect,
            batch_first=True
        )
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)
    
    def init_hidden(self, batch):
        """Initialize hidden states."""
        weight = next(self.parameters()).data
        hid_shape = (self.rnn_layer * self.ndirections, batch, self.hidden_dim)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())
        
    
    def forward_all(self, batch):
        """Return the whole results."""
        self.rnn.flatten_parameters()
        hidden = self.init_hidden(batch.size(0))
        output, hidden = self.rnn(batch, hidden)
        return output
    
    def forward(self, batch):
        """Return the result of the last time step."""
        output = self.forward_all(batch)
        if self.ndirections == 1: return output[:, -1]
        
        forward = output[:, -1, :self.hidden_dim]
        backward = output[:, 0, self.hidden_dim:]
        return torch.cat((forward, backward), dim=1)


class PretrainedWordEmbedding(nn.Module):
    """
    Pre-trained word embedding module.
    """
    def __init__(self, vocab_path: str, device: str):
        """
        vocab_path: path for loading pre-trained word vectors
        device: device
        """
        super().__init__()
        with open(vocab_path) as f:
            lines = f.readlines()
        
        self.device = device
        self.vocab_dim = len(lines[0].split())-1
        self.vocab_len = len(lines) + 4 # vocabulary size = GloVe vocabulary + <oov> + <start> + <end> + <pad>
        vocab = np.zeros((self.vocab_len, self.vocab_dim))
        for i, line in enumerate(tqdm(lines, desc='prepare vocabulary')):
            vocab[i,:] = np.asarray(line.split()[1:], "float32") # save pre-trained vectors
        self.vocab = torch.Tensor(vocab)

    def forward(self, s):
        """
        Input:
            s: [batch, s_len]

        Output:[batch, s_len, vocab_dim]
        """
        batch, s_len = s.size()
        output = torch.zeros(batch, s_len, self.vocab_dim)
        for i in range(batch):
            output[i,:,:] = self.vocab[s[i,:]]

        return output.to(self.device)

    
class CaptionEmbedding(nn.Module):
    """
    Caption embedding module mentioned in 'Generating Question Relevant Captions to Aid Visual Question Answering'.
    This module would consider the question-attended visual feature, question features, and hidden states of word embeddings.
    """
    def __init__(   self,
                    v_dim: int,
                    q_dim: int,
                    hidden_dim: int,
                    max_len: int,
                    device: str,
                    dropout: float = 0.2,
                    rnn_type: str='GRU',

    ):
        """Input:
            v_dim: dimension of the visual feature
            q_dim: dimension of the question feature
            hidden_dim: dimension of the hidden states
            max_len: the maximal length of input captions
            dropout: dropout (default=0.5)
            rnn_type: choose the type of RNN (default=GRU)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.rnn_type = rnn_type
        self.device = device
        
        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.word_rnn = rnn_cls(input_size=q_dim, hidden_size = hidden_dim)
        self.caption_rnn = rnn_cls(input_size=hidden_dim, hidden_size = hidden_dim)
        
        # prepare caption attention module
        self.attention = CaptionAttention(v_dim=v_dim, q_dim=q_dim, hidden_dim=hidden_dim, dropout=dropout)

        # fully-connected layer
        self.fcnet = FCNet(hidden_dim, hidden_dim)

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        hid_shape = (batch, self.hidden_dim)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def select_hidden(self, h, batch):
        if self.rnn_type == 'LSTM':
            return (h[0][:batch], h[1][:batch])
        else:
            return h[:batch]

    def forward(self, v, q, caption, cap_len):
        """Input:
            v: visual feature [batch, v_dim]
            q: question embedding [batch, q_dim]
            caption: caption embedding [batch, c_len, c_dim]
            c_len: lengths for each input caption [batch, 1]
        """
        # Sort input data by decreasing lengths, so that we can process only valid time steps, i.e., no need to process the <pad>
        cap_len, sort_id = cap_len.sort(dim=0, descending=True)
        caption = caption[sort_id]

        # Initialize LSTM states
        batch = caption.size(0)
        h1 = self.init_hidden(batch) # [batch, hidden_dim]
        h2 = self.init_hidden(batch) # [batch, hidden_dim]

        # Create tensor to hold the caption embedding after all time steps
        output = torch.zeros(batch, self.max_len, self.hidden_dim).to(self.device)
        alphas = torch.zeros(batch, self.max_len, self.hidden_dim).to(self.device)

        # This list is for saving the batch size for each time step
        batches = []
        # For each time step:
        for t in range(max(cap_len)):
            # Only generate captions which is longer than t(ignor <pad>)
            batch_t = sum([l > t for l in cap_len])
            batches.append(batch_t)

            # 1: Word RNN
            # Input: input = [batch_t, q_dim], h1 = [batch_t, hidden_dim]
            # Output: [batch_t, hidden_dim]
            h1 = self.select_hidden(h1, batch_t)
            h1 = self.word_rnn(caption[:batch_t, t, :], h1)

            # Attention
            if self.rnn_type == 'LSTM': h1 = h1[0]
            att = self.attention(h1, v, q) # [batch_t, hidden_dim]

            # 2: Caption RNN
            # Input = att_c: [batch_t, hidden_dim], h2 = batch_t, hidden_dim
            # Output = h2: [batch_t, hidden_dim]
            h2 = self.select_hidden(h2, batch_t)
            h2 = self.caption_rnn(att*caption[:batch_t, t, :], h2)

            # Fully-connected layer
            h2 = self.fcnet(h2)

            # Save the results
            output[:batch_t, t, :] = h2
            alphas[:batch_t, t, :] = att

        # Element-wise max pooling

        return output, alphas
