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

    cand = []
    for i in range(len(g) - 1):
        for j in range(i + 1, len(g)):
            if set_id[i] != set_id[j] or i == j:
                continue
            cand.append('%d %d' % (i, j))
    
    if cmd_args.rand_att_type == 'random':
        added = np.random.choice(cand, num_added)
        added = [(int(w.split()[0]), int(w.split()[1])) for w in added]
        g.add_edges_from(added)
        return S2VGraph(g, s2v_g.label)
    elif cmd_args.rand_att_type == 'exhaust':
        g_list = []
        for c in cand:
            x, y = [int(w) for w in c.split()]
            g2 = g.copy()
            g2.add_edge(x, y)
            g_list.append(S2VGraph(g2, s2v_g.label))
        _, _, acc = model(g_list)
        ans = g_list[0]
        for i in range(len(g_list)):
            if acc.numpy()[i] < 1:
                ans = g_list[i]
                break
        return ans
    else:
        raise NotImplementedError

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