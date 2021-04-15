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
import cPickle as cp
sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args, save_args
from dnn import GraphClassifier
from graph_embedding import S2VGraph

sys.path.append('%s/../data_generator' % os.path.dirname(os.path.realpath(__file__)))
from data_util import load_pkl

from graph_common import loop_dataset, load_er_data

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)    
    
    label_map, train_glist, test_glist = load_er_data()    
        
    if cmd_args.saved_model is not None and cmd_args.saved_model != '':        
        print('loading model from %s' % cmd_args.saved_model)
        with open('%s-args.pkl' % cmd_args.saved_model, 'rb') as f:
            base_args = cp.load(f)
        classifier = GraphClassifier(label_map, **vars(base_args))            
        classifier.load_state_dict(torch.load(cmd_args.saved_model + '.model'))
    else:
        classifier = GraphClassifier(label_map, **vars(cmd_args))

    if cmd_args.ctx == 'gpu':
        classifier = classifier.cuda()
    if cmd_args.phase == 'test':
        test_loss = loop_dataset(test_glist, classifier, list(range(len(test_glist))), epoch=101)
        print('\033[93maverage test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

    if cmd_args.phase == 'train':
        optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

        train_idxes = list(range(len(train_glist)))
        best_loss = None
        for epoch in range(cmd_args.num_epochs):
            random.shuffle(train_idxes)
            avg_loss = loop_dataset(train_glist, classifier, train_idxes, optimizer=optimizer, epoch=epoch)
            print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
            
            test_loss = loop_dataset(test_glist, classifier, list(range(len(test_glist))), epoch=epoch)
            print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

            if best_loss is None or test_loss[0] < best_loss:
                best_loss = test_loss[0]
                print('----saving to best model since this is the best valid loss so far.----')
                torch.save(classifier.state_dict(), cmd_args.save_dir + '/epoch-best.model')
                save_args(cmd_args.save_dir + '/epoch-best-args.pkl', cmd_args)
