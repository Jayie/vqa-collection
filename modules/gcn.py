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
    def __init__(self, in_dim, out_dim, num_labels, bias=True):
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


class DirectedGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_labels, dir_num=2):
        # super().__init__(in_dim, out_dim, num_labels, True)
        super().__init__()
        # TODO: Define weights for different <i,j>, <j,i> and <i,i>
        self.out_dim = out_dim
        self.dir_num = dir_num
        weight = []
        for i in range(dir_num):
            weight.append(nn.Linear(in_dim, out_dim, bias=False))
        self.weight = nn.ModuleList(weight)
        # Define biases for different labels
        self.bias = Parameter(torch.FloatTensor(num_labels, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_dim)
        # self.weight_triu.data.uniform_(-stdv, stdv)
        # self.weight_tril.data.uniform_(-stdv, stdv)
        # for i in range(self.dir_num):
        #     self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # def forward(self, feature, graph):
    #     """Input:
    #         feature: [batch, num_objs, in_dim]
    #         graph: [batch, num_objs, num_objs]
    #     Output: [batch, num_objs, out_dim]
    #     """
    #     batch = feature.size(0)
    #     adj = (graph!=0).float()
    #     output = torch.bmm(feature, self.weight.unsqueeze(0).repeat(batch,1,1))
    #     output = torch.bmm(adj, output)
        
    #     # Add bias according to labels
    #     # Need to add the original feature since the diagonal of our relation graph is zero
    #     return feature + output + self.bias[graph.cpu().numpy(),:].sum(2)
    
    def conv(self, feature, graph):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        batch = feature.size(0)
        adj = (graph!=0).float()
        output = torch.zeros_like(feature)
        output += feature
        for i in range(self.dir_num):
            output += self.weight[i](feature)
        
        # Add bias according to labels
        # Need to add the original feature since the diagonal of our relation graph is zero
        return output + self.bias[graph.cpu().numpy(),:].sum(2)
    
    def forward(self, feature, graph):
        return self.conv(feature, graph)


class CorrelatedGraphConv(DirectedGraphConv):
    def __init__(self, in_dim, out_dim, num_labels):
        super().__init__(in_dim, out_dim, num_labels)
        self.dot_product = DotProduct(in_dim, in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def relation_alpha(self, feature, adj):
        # Compute correlations between vi and vj for all vi, vj in input
        alpha = self.dot_product(feature, feature) # [batch, num_objs, num_objs]
        # alpha = max(0, alpha)
        alpha[alpha<0] = 0
        # Only keep the correlation score of neighbors
        alpha = torch.bmm(adj, alpha)
        # Normalize
        alpha = self.softmax(alpha)
        return alpha

    # def forward(self, feature, graph, get_alpha):
    #     """Input:
    #         feature: [batch, num_objs, in_dim]
    #         graph: [batch, num_objs, num_objs]
    #     Output: [batch, num_objs, out_dim]
    #     """
    #     batch = feature.size(0)
    #     adj = (graph!=0).float()
    #     output = torch.bmm(feature, self.weight.unsqueeze(0).repeat(batch,1,1))
    #     output = torch.bmm(adj, output)

    #     # Compute correlations
    #     alpha = self.relation_alpha(feature, adj)
    #     # Mutiply
    #     output = torch.bmm(alpha, output)

    #     # Add bias according to labels
    #     # Need to add the original feature since the diagonal of our relation graph is zero
    #     output = feature + output + self.bias[graph.cpu().numpy(),:].sum(2)
    #     if get_alpha: return output, alpha
    #     else: return output

    def forward(self, feature, graph, get_alpha):
        """Input:
        feature: [batch, num_objs, in_dim]
        graph: [batch, num_objs, num_objs]
        Output: [batch, num_objs, out_dim]
        """
        adj = (graph!=0).float()
        output = self.conv(feature, graph)
        
        # Compute correlations
        alpha = self.relation_alpha(feature, adj)
        
        # Add bias according to labels
        # Need to add the original feature since the diagonal of our relation graph is zero
        output = torch.bmm(alpha, output)
        if get_alpha: return output, alpha
        else: return output


class GCN(nn.Module):
    """
    Relation Encoder mentioned in 'Exploring Visual Relationship for Image Captioning'
    This GCN-based module learns visual features considering relationships.
    """
    def __init__( self,
                  in_dim: int,
                  out_dim: int,
                  num_labels: int,
                  device: str,
                  conv_layer: int = 1,
                  conv_type: str = 'corr',
                  dropout: float = 0.5,
                ):
        super().__init__()
        GraphConv = get_graph_conv(conv_type)
        
        self.gcn = [GraphConv(in_dim, out_dim, num_labels).to(device)]
        for _ in range(conv_layer-1):
            self.gcn.append(GraphConv(out_dim, out_dim, num_labels).to(device))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        output = []
        for layer in self.gcn:
            output.append(layer.__repr__())
        return '\n'.join(output)

    def forward(self, feature, graph, get_alpha):
        """Input:
            feature: [batch, num_objs, in_dim]
            graph: [batch, num_objs, num_objs]
        """
        alphas = []
        for i in range(len(self.gcn)):
            if get_alpha:
                feature, alpha = self.gcn[i](feature, graph, get_alpha)
                alphas.append(alpha)
            else:
                feature = self.gcn[i](feature, graph, get_alpha)
            feature = self.dropout(feature)
            feature = nn.functional.relu(feature)
        
        if get_alpha: return feature, alphas
        else: return feature