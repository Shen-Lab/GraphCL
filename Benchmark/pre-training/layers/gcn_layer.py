import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
    
# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
# 将节点表示h作为信息发出
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """对所有邻节点节点特征求平均并覆盖原本的节点特征。"""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_dim != out_dim:
            self.residual = False
        # in_dim = 128, out_dim = 128
        self.apply_mod = NodeApplyModule(in_dim, out_dim)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        # RELU
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
    '''
    input:

    g: batch_graphs:
        DGLGraph(num_nodes=562, num_edges=2172,
            ndata_schemes={'feat': Scheme(shape=(18,), dtype=torch.float32)}
            edata_schemes={'feat': Scheme(shape=(18,), dtype=torch.float32)})

    h: batch_graphs.ndata['feat']
        torch.Size([562, 128])

    snorm_n: batch_snorm_n
        torch.Size([562, 1])
    
    '''
    def forward(self, g, feature, snorm_n): # feature = torch.Size([562, 128])
        # pdb.set_trace()
        h_in = feature   # to be used for residual connection
        g.ndata['h'] = feature
        g.update_all(msg, reduce) # send message and update nodes
        g.apply_nodes(func=self.apply_mod) # 
        h = g.ndata['h'] # result of graph convolution
        
        if self.graph_norm:
            h = h * snorm_n # normalize activation w.r.t. graph size
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
            
        '''0417 remove'''
        h = self.activation(h)
        if self.residual:
            h = h_in + h # residual connection

        h = self.dropout(h)
        return h


    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)