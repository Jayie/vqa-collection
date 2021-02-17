import math
import numpy as np
from tqdm import tqdm

import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.weight_norm import weight_norm

class FCNet(nn.Module):
    """
    Non-linear fully-connected network.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 mid_dim: int = 0,
                 layer: int = 1,
                 dropout: float = 0,
        ):
        """Input:
            in_dim: input dimension
            out_dim: output dimension
            mid_dim: dimension of the layers in the middle of the network (default = 0)
            layer: number of layers (default = 1)
            dropout: dropout (default = 0, i.e. no use of dropout)
        """

        super().__init__()
        layers = []

        if layer == 1 or mid_dim == 0:
            # If 1-layer:
            # Layer 1: in_dim -> out_dim
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        else:
            # Else:
            # Suppose there are N layers
            # Layer 1: in_dim -> mid_dim
            layers.append(weight_norm(nn.Linear(in_dim, mid_dim), dim=None))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout, inplace=True))
            
            # Layer 2 ~ N-1: mid_dim -> mid_dim
            for _ in range(layer-2):
                layers.append(weight_norm(nn.Linear(mid_dim, mid_dim), dim=None))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout, inplace=True))
            
            # Layer n: mid_dim -> out_dim
            layers.append(weight_norm(nn.Linear(mid_dim, out_dim), dim=None))
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class LReLUNet(nn.Module):
    """
    Fully-connected layer with Leaky ReLU used in 'Generating Question Relevant Captions to Aid Visual Question Answering'
    , which is
        f(x) = LReLU(Wx + b)
    with input features x and ignore the notation of weights and biases for simplicity.
    """
    def __init__(self, in_dim: int, out_dim: int, neg_slope: float = 0.01):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.LeakyReLU(neg_slope)
        )

    def forward(self, x):
        return self.main(x)

class SentenceEmbedding(nn.Module):
    """
    Sentence embedding module.
    """
    def __init__(   self,
                    in_dim: int,
                    hidden_dim: int,
                    device: str,
                    rnn_layer: int = 1,
                    dropout: float = 0.5,
                    rnn_type: str = 'LSTM',
                    bidirect: bool = False,
        ):
        """Input:
            in_dim: input dimension (i.e. dimension of word embedding)
            hidden_dim: dimension of the hidden state
            rnn_layer: number of RNN layers (default = 1)
            dropout: dropout (default = 0.5)
            rnn_type: choose the type of RNN (default =LSTM)
            bidirect: if True, use a bidirectional RNN (default = False)
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
        self.device = device
    
    def init_hidden(self, batch):
        """Initialize hidden states."""
        shape = (self.rnn_layer * self.ndirections, batch, self.hidden_dim)
        if self.rnn_type == 'LSTM':
            return (torch.zeros(shape).to(self.device),
                    torch.zeros(shape).to(self.device))
        else:
            return torch.zeros(shape).to(self.device)
        
    
    def forward_all(self, batch):
        """Return the whole results."""
        hidden = self.init_hidden(batch.size(0))
        self.rnn.flatten_parameters()
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


class CaptionAttention(nn.Module):
    """
    Caption attention module for caption embedding mentioned in 'Generating Question Relevant Captions to Aid Visual Question Answering'
    , which is
        a = sigmoid(h * f(v) + h * f(q)),
    where h is the the hidden state of the Word GRU, v is the question-attended visual feature, q is the question embedding,
    and f denotes a fully-connected layers.
    """
    def __init__(self,
                 v_dim: int,
                 q_dim: int,
                 hidden_dim: int,
                 neg_slope: float = 0.01,
                 dropout: float = 0.2,
    ):
        super().__init__()
        self.W_v = LReLUNet(v_dim, hidden_dim, neg_slope)
        self.W_q = LReLUNet(q_dim, hidden_dim, neg_slope)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def logits(self, h, q):
        """Input:
            h: [batch, hidden_dim]
            q: [batch, hidden_dim]
        """
        # element-wise multiply each visual features with question features
        joint = h * q
        joint = self.dropout(joint)
        return joint # [batch, h_len, hidden_dim]
    
    def forward(self, h, v, q):
        """Input:
            h: [batch, hidden_dim]
            v: [batch, v_dim]
            q: [batch, q_dim]
        Output: [batch, hidden_dim]
        """
        v = self.W_v(v) # [batch, hidden_dim]
        q = self.W_q(q) # [batch, hidden_dim]
        h = self.logits(h, v) + self.logits(h, q) # [batch, hidden_dim]
        return self.sigmoid(h)

    
class CaptionEmbedding(nn.Module):
    """
    Caption embedding module mentioned in 'Generating Question Relevant Captions to Aid Visual Question Answering'
    This module would consider the question-attended visual feature, question features, and hidden states of word embeddings.
    """
    def __init__(   self,
                    v_dim: int,
                    q_dim: int,
                    c_dim: int,
                    hidden_dim: int,
                    max_len: int,
                    device: str,
                    dropout: float = 0.2,
                    neg_slope: float = 0.01,
                    rnn_type: str='GRU',

    ):
        """Input:
            v_dim: dimension of the visual feature
            q_dim: dimension of the question feature
            c_dim: dimension of the caption embedding
            hidden_dim: dimension of the hidden states
            max_len: the maximal length of input captions
            dropout: dropout (default = 0.5)
            neg_slope: negative slope for Leaky ReLU (default = 0.01)
            rnn_type: choose the type of RNN (default = GRU)
        """
        super().__init__()
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.rnn_type = rnn_type
        self.device = device
        
        # Prepare word embedding layer and sentence embedding layer.
        # Since we need to compute the attention for each time step, we use RNN cells here.
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTMCell if rnn_type =='LSTM' else nn.GRUCell
        self.word_rnn = rnn_cls(input_size=c_dim, hidden_size=c_dim)
        self.caption_rnn = rnn_cls(input_size=c_dim, hidden_size=hidden_dim)
        
        # prepare caption attention module
        self.attention = CaptionAttention(v_dim=v_dim, q_dim=q_dim, hidden_dim=c_dim, dropout=dropout)
        # fully-connected layer
        self.fcnet = LReLUNet(hidden_dim, hidden_dim, neg_slope)
        # max-pooling layer
        self.maxpool = nn.MaxPool1d(max_len)

    def init_hidden(self, shape):
        """Initialize hidden states."""
        if self.rnn_type == 'LSTM':
            return (torch.zeros(shape).to(self.device), torch.zeros(shape).to(self.device))
        else:
            return torch.zeros(shape).to(self.device)

    def select_hidden(self, h, batch):
        if self.rnn_type == 'LSTM':
            return (h[0][:batch], h[1][:batch])
        else:
            return h[:batch]

    def forward(self, v, q, caption, cap_len):
        """Input:
            v: visual features [batch, v_dim]
            q: question embedding [batch, q_dim]
            caption: caption embedding [batch, c_len, c_dim]
            c_len: lengths for each input caption [batch, 1]
        """

        # Sort input data by decreasing lengths, so that we can process only valid time steps, i.e., no need to process the <pad>
        cap_len, sort_id = cap_len.sort(dim=0, descending=True)
        caption = caption[sort_id]
        v = v[sort_id]
        restore_id = sorted(sort_id, key=lambda k: sort_id[k]) # to restore the order

        # Initialize RNN states
        batch = caption.size(0)
        h1 = self.init_hidden((batch, self.c_dim)) # [batch, c_dim]
        h2 = self.init_hidden((batch, self.hidden_dim)) # [batch, hidden_dim]

        # Create tensor to hold the caption embedding after all time steps
        output = torch.zeros(batch, self.hidden_dim, self.max_len).to(self.device)
        alphas = torch.zeros(batch, self.max_len, self.c_dim).to(self.device)

        # This list is for saving the batch size for each time step
        batches = []
        # For each time step:
        for t in range(max(cap_len)):
            # Only encode captions which is longer than t (ignore <pad>)
            batch_t = sum([l > t for l in cap_len])
            batches.append(batch_t)
            h1 = self.select_hidden(h1, batch_t)
            h2 = self.select_hidden(h2, batch_t)

            # 1: Word RNN
            # Input = caption: [batch_t, q_dim], h1: [batch_t, c_dim]
            # Output = h1: [batch_t, c_dim]
            h1 = self.word_rnn(caption[:batch_t, t, :], h1)
            h = h1[0] if self.rnn_type == 'LSTM' else h1

            # Attention
            att = self.attention(h, v[:batch_t, :], q[:batch_t, :]) # [batch_t, c_dim]
            att = att * caption[:batch_t, t, :] # [batch_t, c_dim]

            # 2: Caption RNN
            # Input = att_c: [batch_t, c_dim], h2 = [batch_t, hidden_dim]
            # Output = h2: [batch_t, hidden_dim]
            h2 = self.caption_rnn(att, h2)
            h = h2[0] if self.rnn_type == 'LSTM' else h2

            # Fully-connected layer
            h = self.fcnet(h)

            # Save the results
            output[:batch_t, :, t] = h
            alphas[:batch_t, t, :] = att
        # Element-wise max pooling
        output = self.maxpool(output).squeeze() # [batch, hidden_dim]

        return output[restore_id,:], alphas[restore_id,:,:]


class DotProduct(nn.Module):
    def __init__(self, a_dim, b_dim, out_dim):
        self.wa = nn.Linear(a_dim, out_dim)
        self.wb = nn.Linear(b_dim, out_dim)

    def forward(self, a, b):
        """
        a: [batch, a_len, a_dim]
        b: [batch, b_len, b_dim]
        output: [batch, a_len, b_len]
        """
        a = self.wa(a)
        b = self.wb(b)
        b = torch.transpose(b, 1, 2)
        return torch.bmm(a, b)


class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    reference: https://github.com/tkipf/pygcn
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init_()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.dot_product = DotProduct(in_dim, in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, graph):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        adj = (graph != 0) # Adjacency matrix
        # Compute similarity between vi and vj for all vi, vj in input
        alpha = self.dot_product(feature, feature) # [batch, num_objs, num_objs]
        # Keep alpha >= 0
        alpha[alpha < 0] = 0
        # Multiply alpha and adjacency matrix since we need only the relations of neighbors
        alpha = torch.mm(alpha, adj)
        # Normalize
        alpha = self.softmax(alpha)
        
        # TODO: update the features considering alpha
        # TODO: consider the relation type

        # Original code
        # support = torch.mm(feature, self.weight)
        # output = torch.spmm(adj, support)
        # if self.bias is not None: return output + self.bias
        # else: return output


class RelationEncoder(nn.Module):
    """
    Relation Encoder mentioned in 'Exploring Visual Relationship for Image Captioning'
    This GCN-based module learns visual features considering relationships.
    """
    def __init__( self,
                  in_dim: int,
                  out_dim: int,
                  relation_num: int,
                  conv_layer: int = 1,
                ):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        return