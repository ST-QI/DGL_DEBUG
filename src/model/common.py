import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATConv

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # def forward(self, inpu, adj):
    def forward(self, inpu):
        support = torch.matmul(inpu, self.weight)
        #print(adj.size())
        #print(support.size())
        # output = torch.matmul(adj, support)
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class ResGCN(nn.Module):
    def __init__(self, features, adj):
        super(ResGCN, self).__init__()
        self.adj = adj
        self.A = nn.Parameter(torch.from_numpy(self.adj).float())
        self.relu = nn.LeakyReLU(0.2)
        self.graph_conv1 = GraphConvolution(features, features)
        self.graph_conv2 = GraphConvolution(features, features)

    def forward(self, inpu):
        adj = gen_adj(self.A).detach()
        res_g = self.graph_conv1(inpu, adj)
        res_g = self.relu(res_g)
        res_g = self.graph_conv1(res_g, adj)
        return inpu + res_g

class ChannelSpatialGATLayer(nn.Module):
    def __init__(self,
                 cha_nodes_n=64,
                 cha_in_dim=36,
                 cha_out_dim=6,
                 cha_num_heads=6,
                 spa_nodes_n=36,
                 spa_in_dim=64,
                 spa_out_dim=8,
                 spa_num_heads=8,
                 ):
        super(ChannelSpatialGATLayer, self).__init__()
        self.GAT_body_channel = Graph_Attention_Convolution(nodes_n=cha_nodes_n,
                                                            in_dim=cha_in_dim,
                                                            out_dim=cha_out_dim,
                                                            num_heads=cha_num_heads)
        self.GAT_body_spatial = Graph_Attention_Convolution(nodes_n=spa_nodes_n,
                                                            in_dim=spa_in_dim,
                                                            out_dim=spa_out_dim,
                                                            num_heads=spa_num_heads)

    def forward(self, input):
        ndata, cha_con, spa_con = input
        # ndata(b,6,6,64,36)
        x = ndata.permute(3,0,1,2,4)  # x(64,b,6,6,36)
        x = self.GAT_body_channel(cha_con, x)  # x(64,b,6,6,36)
        x = x.permute(4,1,2,3,0)  # x(36,b,6,6,64)
        x = self.GAT_body_spatial(spa_con, x)  # x(36,b,6,6,64)
        x = x.permute(1,2,3,4,0)  # x(b,6,6,64,36)
        return x, cha_con, spa_con

class Graph_Attention_Convolution(nn.Module):
    def __init__(self,
                 nodes_n=64,
                 in_dim=36,
                 out_dim=6,
                 num_heads=6,
                 ):
        super(Graph_Attention_Convolution, self).__init__()

        self.aggregator = AttentionAggregator(nodes_n, in_dim, out_dim, num_heads)

    def forward(self, connectivity, nodes_data):
        out = self.aggregator(connectivity, nodes_data)
        return out

class AttentionAggregator(nn.Module):
    def __init__(self, nodes_n=64, in_dim=36, out_dim=6, num_heads=6):
        super(AttentionAggregator, self).__init__()
        self.nodes_n = nodes_n
        self.mh_gat = GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=num_heads, allow_zero_in_degree=True)

    def forward(self, connectivity, nodes_data):
        edges = (connectivity[0][0], connectivity[1][0])
        g = dgl.graph(edges, num_nodes=self.nodes_n)
        with g.local_scope():
            emb = self.mh_gat(g, nodes_data)
            emb = emb.flatten(4)
        del g
        return emb

