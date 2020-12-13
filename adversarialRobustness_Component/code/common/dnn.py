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

from cmd_args import cmd_args
from graph_embedding import EmbedMeanField, EmbedLoopyBP

sys.path.append('%s/../../pytorch_structure2vec/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)
        
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()
        self.hidden_size = hidden_size
        if hidden_size > 0:
            self.h1_weights = nn.Linear(input_size, hidden_size)
            self.last_weights = nn.Linear(hidden_size, num_class)
        else:
            self.last_weights = nn.Linear(input_size, num_class)

        weights_init(self)

    def forward(self, x, y = None):
        if self.hidden_size:
            x = self.h1_weights(x)
            x = F.relu(x)
        
        logits = self.last_weights(x)
        logits = F.log_softmax(logits, dim=1)        

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu()
            return logits, loss, acc
        else:
            return logits

class GraphClassifier(nn.Module):
    def __init__(self, label_map, **kwargs):
        super(GraphClassifier, self).__init__()
        self.label_map = label_map
        if kwargs['gm'] == 'mean_field':
            model = EmbedMeanField
        elif kwargs['gm'] == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % kwargs['gm'])
            sys.exit()
        if 'feat_dim' in kwargs:
            self.feat_dim = kwargs['feat_dim']
        else:
            self.feat_dim = 0
        self.s2v = model(latent_dim=kwargs['latent_dim'], 
                        output_dim=kwargs['out_dim'],
                        num_node_feats=kwargs['feat_dim'], 
                        num_edge_feats=0,
                        max_lv=kwargs['max_lv'])
        out_dim = kwargs['out_dim']
        if out_dim == 0:
            out_dim = kwargs['latent_dim']
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=kwargs['hidden'], num_class=len(label_map))

        self.projection_head = nn.Sequential(nn.Linear(kwargs['latent_dim'], kwargs['latent_dim']), nn.ReLU(inplace=True), nn.Linear(kwargs['latent_dim'], kwargs['latent_dim']))


    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        concat_feat = []
        for i in range(len(batch_graph)):
            labels[i] = self.label_map[batch_graph[i].label]
            n_nodes += batch_graph[i].num_nodes
            if batch_graph[i].node_tags is not None:
                concat_feat += batch_graph[i].node_tags
        if len(concat_feat):
            node_feat = torch.zeros(n_nodes, self.feat_dim)
            concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
            node_feat.scatter_(1, concat_feat, 1)
        else:
            node_feat = torch.ones(n_nodes, 1)
        if cmd_args.ctx == 'gpu':
            node_feat = node_feat.cuda()
        return node_feat, None, labels

    def forward(self, batch_graph): 
        node_feat, edge_feat, labels = self.PrepareFeatureLabel(batch_graph)
        if cmd_args.ctx == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            
        _, embed = self.s2v(batch_graph, node_feat, edge_feat, pool_global=True)

        return self.mlp(embed, labels)


    def forward_cl(self, batch_graph): 
        node_feat, edge_feat, labels = self.PrepareFeatureLabel(batch_graph)
        if cmd_args.ctx == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            
        _, embed = self.s2v(batch_graph, node_feat, edge_feat, pool_global=True)
        embed = self.projection_head(embed)

        return embed

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.matmul(x1, x2.t()) / torch.matmul(x1_abs.view(-1, 1), x2_abs.view(1, -1))
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss











