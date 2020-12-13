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
from copy import deepcopy

from q_net import NStepQNet, QNet, greedy_actions
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from graph_embedding import S2VGraph

from rl_common import GraphEdgeEnv, load_graphs, test_graphs, attackable, load_base_model

sys.path.append('%s/../graph_classification' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset, load_er_data

def propose_attack(model, s2v_g, num_added=1):
    g = s2v_g.to_networkx()
    comps = [c for c in nx.connected_component_subgraphs(g)]
    set_id = {}

    for i in range(len(comps)):
        for j in comps[i].nodes():
            set_id[j] = i

    node_feat, edge_feat, labels = model.PrepareFeatureLabel([s2v_g])
    if cmd_args.ctx == 'gpu':
        node_feat = node_feat.cuda()
        labels = labels.cuda()

    cand_list = [s2v_g]
    for l in range( len(model.label_map) ):
        if l == s2v_g.label:
            continue
        labels[0] = l
        model.zero_grad()
        (_, embed), sp_dict = model.s2v([s2v_g], node_feat, edge_feat, pool_global=True, n2n_grad=True)
        _, loss, _ = model.mlp(embed, labels)
        loss.backward()
        grad = sp_dict['n2n'].grad.data.numpy().flatten()    
        idxes = np.argsort(grad)
        added = []

        for p in idxes:
            x = p // s2v_g.num_nodes
            y = p % s2v_g.num_nodes
            if set_id[x] != set_id[y] or x == y or grad[p] > 0:
                continue
            added.append((x, y))
            if len(added) >= num_added:
                break
        if len(added) == 0:
            continue
        g2 = g.copy()
        g2.add_edges_from(added)

        cand_list.append( S2VGraph(g2, s2v_g.label) )
    
    _, _, acc = model(cand_list)
    acc = acc.double().cpu().numpy()
    for i in range(len(cand_list)):
        if acc[i] < 1.0:
            return cand_list[i]
    return cand_list[0]

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    label_map, train_glist, test_glist = load_er_data()

    base_classifier = load_base_model(label_map, test_glist)

    new_test_list = []
    for g in tqdm(test_glist):
        new_test_list.append(propose_attack(base_classifier, g))

    test_graphs(base_classifier, new_test_list)