import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from util.attention import DotProduct

class BaseGraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    reference: https://github.com/tkipf/pygcn
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init_()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
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
            graph: [batch, num_objs, num_objs] (note that this graph is an adjacency matrix)
        Output: [batch, num_objs, out_dim]
        """
        output = torch.mm(feature, self.weight)
        output = torch.mm(graph, feature)
        if self.bias is not None: return output + self.bias
        return output


class DirectedGraphConv(BaseGraphConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, False)
        # TODO: define biases for different labels

    def forward(self, feature, graph):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        output = torch.mm(feature, self.weight)
        # Add bias for certain labels
        # TODO
        return output


class CorrelatedGraphConv(DirectedGraphConv):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.dot_product = DotProduct(in_dim, in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, graph):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        output = torch.mm(feature, self.weight)

        # Compute correlations between vi and vj for all vi, vj in input
        alpha = self.dot_product(feature, feature) # [batch, num_objs, num_objs]
        # alpha = max(0, alpha)
        alpha[alpha<0] = 0
        # Only keep the correlation score of neighbors
        alpha = torch.mm(alpha, graph!=0)
        # Normalize
        alpha = self.softmax(alpha)
        # Mutiply
        output = torch.mm(alpha, output)

        # Add bias for certain labels
        return output