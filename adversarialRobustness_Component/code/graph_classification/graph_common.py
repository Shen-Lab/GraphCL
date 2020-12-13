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

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from graph_embedding import S2VGraph

sys.path.append('%s/../data_generator' % os.path.dirname(os.path.realpath(__file__)))
from data_util import load_pkl

from copy import deepcopy


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size, epoch=0):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]

        if epoch <= 100:
            # if False:
            batch_graph_aug = [deepcopy(g).node_dropping() for g in batch_graph]
            x1 = classifier.forward_cl(batch_graph)
            x2 = classifier.forward_cl(batch_graph_aug)
            loss = classifier.loss_cl(x1, x2)
            acc = torch.zeros(1)
        else:
            _, loss, acc = classifier(batch_graph)
        
        acc = acc.sum() / float(acc.size()[0])
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()

        loss = loss.data.cpu().numpy()[0]
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss

def load_er_data():
    frac_train = 0.9
    pattern = 'nrange-%d-%d-n_graph-%d-p-%.2f' % (cmd_args.min_n, cmd_args.max_n, cmd_args.n_graphs, cmd_args.er_p)

    num_train = int(frac_train * cmd_args.n_graphs)

    train_glist = []
    test_glist = []
    label_map = {}
    for i in range(cmd_args.min_c, cmd_args.max_c + 1):
        cur_list = load_pkl('%s/ncomp-%d-%s.pkl' % (cmd_args.data_folder, i, pattern), cmd_args.n_graphs)
        assert len(cur_list) == cmd_args.n_graphs
        train_glist += [S2VGraph(cur_list[j], i) for j in range(num_train)]
        test_glist += [S2VGraph(cur_list[j], i) for j in range(num_train, len(cur_list))]
        label_map[i] = i - cmd_args.min_c
    cmd_args.num_class = len(label_map)
    cmd_args.feat_dim = 1
    print('# train:', len(train_glist), ' # test:', len(test_glist))

    return label_map, train_glist, test_glist

