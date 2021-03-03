import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .modules import DotProduct

def get_graph_conv(conv_type):
    return {
        'base': BaseGraphConv,
        'direct': DirectedGraphConv,
        'corr': CorrelatedGraphConv
    }[conv_type]

class BaseGraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    reference: https://github.com/tkipf/pygcn
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
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
        batch = feature.size(0)
        output = torch.bmm(feature, self.weight.unsqueeze(0).repeat(batch,1,1))
        output = torch.bmm(graph, output)
        if self.bias is not None: return output + self.bias.unsqueeze(0).repeat(batch,1,1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'


class DirectedGraphConv(BaseGraphConv):
    def __init__(self, in_dim, out_dim, num_labels):
        super().__init__(in_dim, out_dim, True)
        # TODO: Define weights for different <i,j>, <j,i> and <i,i>
        # Define biases for different labels
        self.bias = Parameter(torch.FloatTensor(num_labels, out_dim))
        self.reset_parameters()

    def forward(self, feature, graph):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        batch = feature.size(0)
        adj = (graph!=0).float()
        output = torch.bmm(feature, self.weight.unsqueeze(0).repeat(batch,1,1))
        output = torch.bmm(adj, output)
        
        # Add bias according to labels
        return output + self.bias[graph.numpy(),:].sum(2)


class CorrelatedGraphConv(DirectedGraphConv):
    def __init__(self, in_dim, out_dim, num_labels):
        super().__init__(in_dim, out_dim, num_labels)
        self.dot_product = DotProduct(in_dim, in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, graph):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        batch = feature.size(0)
        adj = (graph!=0).float()
        output = torch.bmm(feature, self.weight.unsqueeze(0).repeat(batch,1,1))
        output = torch.bmm(adj, output)

        # Compute correlations between vi and vj for all vi, vj in input
        alpha = self.dot_product(feature, feature) # [batch, num_objs, num_objs]
        # alpha = max(0, alpha)
        alpha[alpha<0] = 0
        # Only keep the correlation score of neighbors
        alpha = torch.bmm(adj, alpha)
        # Normalize
        alpha = self.softmax(alpha)
        # Mutiply
        output = torch.bmm(alpha, output)

        # Add bias according to labels
        return output + self.bias[graph.numpy(),:].sum(2)