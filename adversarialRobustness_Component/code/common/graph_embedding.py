from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import networkx as nx

sys.path.append('%s/../../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB
from pytorch_util import weights_init

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None):        
        self.num_nodes = len(g)
        self.node_tags = node_tags
        x, y = zip(*g.edges())
        self.num_edges = len(x)
        self.label = label
        
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()

    def node_dropping(self):
        node_num = self.num_nodes
        edge_num = self.edge_pairs.shape[0] / 2

        num_drop = int(node_num * 0.2)
        idx_nondrop = np.random.choice(node_num, node_num-num_drop, replace=False)
        idx_nondrop.sort()

        edge_pairs = np.resize(self.edge_pairs, (edge_num, 2))
        adj = np.zeros((node_num, node_num))
        adj[edge_pairs[:, 0], edge_pairs[:, 1]] = 1
        adj[list(range(node_num)), list(range(node_num))] = 1
        edge_pairs = np.array(adj[idx_nondrop, :][:, idx_nondrop].nonzero()[-2:]).transpose().flatten()

        self.num_nodes = node_num - num_drop
        n = edge_pairs.shape[0]
        self.edge_pairs = edge_pairs

        return self

    def to_networkx(self):
        edges = np.reshape(self.edge_pairs, (self.num_edges, 2))
        g = nx.Graph()
        g.add_edges_from(edges)
        return g

class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        if ctx.needs_input_grad[0]:
            grad_matrix1 = Variable(torch.mm(grad_output.data, dense_mat.data.t()))
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)


class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.max_lv = max_lv  

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)        
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)

        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat, pool_global=True, n2n_grad=False, e2n_grad=False): 
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        if type(node_feat) is torch.cuda.FloatTensor:
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp, requires_grad=n2n_grad)
        e2n_sp = Variable(e2n_sp, requires_grad=e2n_grad)
        subg_sp = Variable(subg_sp)

        h = self.mean_field(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, pool_global)

        if n2n_grad or e2n_grad:
            sp_dict = {'n2n' : n2n_sp, 'e2n' : e2n_sp}
            return h, sp_dict
        else:
            return h

    def mean_field(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, pool_global):
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        input_potential = F.relu(input_message)

        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            n2npool = gnn_spmm(n2n_sp, cur_message_layer)
            node_linear = self.conv_params( n2npool )
            merged_linear = node_linear + input_message

            cur_message_layer = F.relu(merged_linear)
            lv += 1
        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = cur_message_layer
        
        if pool_global:
            y_potential = gnn_spmm(subg_sp, reluact_fp)
            return reluact_fp, F.relu(y_potential)
        else:
            return reluact_fp

class EmbedLoopyBP(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
        super(EmbedLoopyBP, self).__init__()
        self.latent_dim = latent_dim
        self.max_lv = max_lv

        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        self.out_params = nn.Linear(latent_dim, output_dim)

        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat): 
        n2e_sp, e2e_sp, e2n_sp, subg_sp = S2VLIB.PrepareLoopyBP(graph_list)
        if type(node_feat) is torch.cuda.FloatTensor:
            n2e_sp = n2e_sp.cuda()
            e2e_sp = e2e_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
        node_feat = Variable(node_feat)
        edge_feat = Variable(edge_feat)
        n2e_sp = Variable(n2e_sp)
        e2e_sp = Variable(e2e_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        h = self.loopy_bp(node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp)
        
        return h

    def loopy_bp(self, node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp):
        input_node_linear = self.w_n2l(node_feat)
        input_edge_linear = self.w_e2l(edge_feat)

        n2epool_input = gnn_spmm(n2e_sp, input_node_linear)
        
        input_message = input_edge_linear + n2epool_input
        input_potential = F.relu(input_message)

        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            e2epool = gnn_spmm(e2e_sp, cur_message_layer)
            edge_linear = self.conv_params(e2epool)                    
            merged_linear = edge_linear + input_message

            cur_message_layer = F.relu(merged_linear)
            lv += 1

        e2npool = gnn_spmm(e2n_sp, cur_message_layer)
        hidden_msg = F.relu(e2npool)
        out_linear = self.out_params(hidden_msg)
        reluact_fp = F.relu(out_linear)

        y_potential = gnn_spmm(subg_sp, reluact_fp)

        return F.relu(y_potential)
