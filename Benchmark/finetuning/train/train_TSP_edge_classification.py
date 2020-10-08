"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl
import time
from train.metrics import binary_f1_score
import pdb


def train_epoch(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_f1 = 0
    nb_data = 0
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
          'Epoch [{:>2d}]: learning rate: [{:.10f}]'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_f1 += binary_f1_score(batch_scores, batch_labels)

        if iter % 10 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Iter [{:>3d}]: Loss [{:.4f}]   Train F1 [{:.4f}]'.format(iter, loss, epoch_train_f1 / (iter + 1)))
                 
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Iter [{:>3d}]: Loss [{:.4f}]   Train F1 [{:.4f}]'.format(iter, loss, epoch_train_f1 / (iter + 1)))
    epoch_loss /= (iter + 1)
    epoch_train_f1 /= (iter + 1)
    return epoch_loss, epoch_train_f1, optimizer


def evaluate_network(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_f1 = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_snorm_n = batch_snorm_n.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_f1 += binary_f1_score(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_f1 /= (iter + 1)
        
    return epoch_test_loss, epoch_test_f1


