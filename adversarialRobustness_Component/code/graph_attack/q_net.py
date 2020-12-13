from __future__ import print_function

import os
import sys
import numpy as np
import torch
import networkx as nx
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append('%s/../../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from graph_embedding import EmbedMeanField, EmbedLoopyBP
from cmd_args import cmd_args
from modules.custom_mod import JaggedMaxModule

from rl_common import local_args

def greedy_actions(q_values, v_p, banned_list):
    actions = []
    offset = 0
    banned_acts = []
    prefix_sum = v_p.data.cpu().numpy()
    for i in range(len(prefix_sum)):
        n_nodes = prefix_sum[i] - offset

        if banned_list is not None and banned_list[i] is not None:
            for j in banned_list[i]:
                banned_acts.append(offset + j)                    
        offset = prefix_sum[i]

    q_values = q_values.data.clone()
    if len(banned_acts):
        q_values[banned_acts, :] = np.finfo(np.float64).min
    jmax = JaggedMaxModule()
    values, actions = jmax(Variable(q_values), v_p)

    return actions.data, values.data
    
class QNet(nn.Module):
    def __init__(self, s2v_module = None):
        super(QNet, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.out_dim == 0:
            embed_dim = cmd_args.latent_dim
        else:
            embed_dim = cmd_args.out_dim
        if local_args.mlp_hidden:
            self.linear_1 = nn.Linear(embed_dim * 2, local_args.mlp_hidden)
            self.linear_out = nn.Linear(local_args.mlp_hidden, 1)
        else:
            self.linear_out = nn.Linear(embed_dim * 2, 1)
        weights_init(self)

        if s2v_module is None:
            self.s2v = model(latent_dim=cmd_args.latent_dim, 
                            output_dim=cmd_args.out_dim,
                            num_node_feats=2,
                            num_edge_feats=0,
                            max_lv=cmd_args.max_lv)
        else:
            self.s2v = s2v_module

    def PrepareFeatures(self, batch_graph, picked_nodes):
        n_nodes = 0
        prefix_sum = []
        picked_ones = []
        for i in range(len(batch_graph)):
            if picked_nodes is not None and picked_nodes[i] is not None:
                assert picked_nodes[i] >= 0 and picked_nodes[i] < batch_graph[i].num_nodes
                picked_ones.append(n_nodes + picked_nodes[i])
            n_nodes += batch_graph[i].num_nodes
            prefix_sum.append(n_nodes)

        node_feat = torch.zeros(n_nodes, 2)
        node_feat[:, 0] = 1.0

        if len(picked_ones):
            node_feat.numpy()[picked_ones, 1] = 1.0
            node_feat.numpy()[picked_ones, 0] = 0.0

        return node_feat, torch.LongTensor(prefix_sum)

    def add_offset(self, actions, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        shifted = []        
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)

        return shifted

    def rep_global_embed(self, graph_embed, v_p):
        prefix_sum = v_p.data.cpu().numpy()

        rep_idx = []        
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            rep_idx += [i] * n_nodes

        rep_idx = Variable(torch.LongTensor(rep_idx))
        if cmd_args.ctx == 'gpu':
            rep_idx = rep_idx.cuda()
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    def forward(self, time_t, states, actions, greedy_acts = False):
        batch_graph, picked_nodes, banned_list = zip(*states)

        node_feat, prefix_sum = self.PrepareFeatures(batch_graph, picked_nodes)
        
        if cmd_args.ctx == 'gpu':
            node_feat = node_feat.cuda()
            prefix_sum = prefix_sum.cuda()
        prefix_sum = Variable(prefix_sum)

        embed, graph_embed = self.s2v(batch_graph, node_feat, None, pool_global=True)

        if actions is None:
            graph_embed = self.rep_global_embed(graph_embed, prefix_sum)
        else:
            shifted = self.add_offset(actions, prefix_sum)
            embed = embed[shifted, :]
        
        embed_s_a = torch.cat((embed, graph_embed), dim=1)

        if local_args.mlp_hidden:
            embed_s_a = F.relu( self.linear_1(embed_s_a) )
        
        raw_pred = self.linear_out(embed_s_a)
        
        if greedy_acts:
            actions, _ = greedy_actions(raw_pred, prefix_sum, banned_list)
            
        return actions, raw_pred, prefix_sum

class NStepQNet(nn.Module):
    def __init__(self, num_steps, s2v_module = None):
        super(NStepQNet, self).__init__()

        list_mod = [QNet(s2v_module)]

        for i in range(1, num_steps):
            list_mod.append(QNet(list_mod[0].s2v))
        
        self.list_mod = nn.ModuleList(list_mod)

        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts = False):
        assert time_t >= 0 and time_t < self.num_steps

        return self.list_mod[time_t](time_t, states, actions, greedy_acts)
