"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import dgl
import train.aug as aug
import pdb
from train.aug import AverageMeter
import time


def train_epoch(model, optimizer, device, data_loader, epoch, drop_percent, temp=0.5, aug_type='nn', head=False):
    
    model.train()
    epoch_loss = 0
    
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
         'Epoch: [{:>2d}]  learning rate: [{:.10f}]'.format(epoch + 1, optimizer.param_groups[0]['lr']))
    t0 = time.time()
   
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        aug_batch_graphs = dgl.unbatch(batch_graphs)
        aug_list1, aug_list2 = aug.aug_double(aug_batch_graphs, aug_type)
        batch_graphs, batch_snorm_n, batch_snorm_e= aug.collate_batched_graph(aug_list1)
        aug_batch_graphs, aug_batch_snorm_n, aug_batch_snorm_e= aug.collate_batched_graph(aug_list2)

        aug_batch_x = aug_batch_graphs.ndata['feat'].to(device)  # num x feat
        aug_batch_e = aug_batch_graphs.edata['feat'].to(device)
        aug_batch_snorm_e = aug_batch_snorm_e.to(device)
        aug_batch_snorm_n = aug_batch_snorm_n.to(device)

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        ori_vector = model.forward(batch_graphs, batch_x, batch_e, 
                                   batch_snorm_n, batch_snorm_e, mlp=False, head=head)

        arg_vector = model.forward(aug_batch_graphs, aug_batch_x, aug_batch_e,
                                   aug_batch_snorm_n, aug_batch_snorm_e, mlp=False, head=head)
        
        sim_matrix_tmp2 = aug.sim_matrix2(ori_vector, arg_vector, temp=temp)
        row_softmax = nn.LogSoftmax(dim=1) 
        row_softmax_matrix = -row_softmax(sim_matrix_tmp2)
        
        colomn_softmax = nn.LogSoftmax(dim=0)
        colomn_softmax_matrix = -colomn_softmax(sim_matrix_tmp2)
        
        row_diag_sum = aug.compute_diag_sum(row_softmax_matrix)
        colomn_diag_sum = aug.compute_diag_sum(colomn_softmax_matrix)
        contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))
        
        contrastive_loss.backward()
        optimizer.step()
        epoch_loss += contrastive_loss.detach().item()

        if iter % 20 == 0:
            if iter == 0:
                tot = time.time() - t0
                t1 = tot
            else:
                t1 = time.time() - t0 - tot
                tot = time.time() - t0
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                 'Aug: [{}]  Epoch: [{}/{}]  Iter: [{}/{}]  Loss: [{:.4f}]  Time Taken: [{:.2f} min]'
                 .format(aug_type, epoch + 1, 80, iter, len(data_loader), contrastive_loss, t1 / 60))
            
    epoch_loss /= (iter + 1)
    print('Epoch: [{:>2d}]  Loss: [{:.4f}]'.format(epoch + 1, epoch_loss))

    return epoch_loss, optimizer
