import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse

class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

    self.sigmoid = nn.Sigmoid()
    self.bceloss = nn.BCELoss()
    self.projection_head = nn.Sequential(nn.Linear(96, 96), nn.ReLU(inplace=True), nn.Linear(96, 96))

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

  def forward_GAE(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    _, M = self.encoder(x, edge_index, batch)
    M = self.projection_head(M)

    C = torch.einsum('ik,jk->ij', M, M)
    C = self.sigmoid(C)
    node_num, _ = C.size()
    adj = torch.eye(node_num)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    adj = adj.to(C.device)

    loss = self.bceloss(C, adj)
    
    return loss


if __name__ == '__main__':
    
    args = arg_parse()
    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
    epochs = 20
    log_interval = 1
    batch_size = 1
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    # dataset = TUDataset(path, name=DS).shuffle()
    dataset = TUDataset(path, name=DS)
    try:
        dataset_num_features = dataset.num_features
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataloader_v = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GcnInfomax(args.hidden_dim, args.num_gc_layers).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    
    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader)
    res = evaluate_embedding(emb, y)
    accuracies['logreg'].append(res[0])
    accuracies['svc'].append(res[1])
    accuracies['linearsvc'].append(res[2])
    accuracies['randomforest'].append(res[3])


    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:

            data = data.to(device)
            optimizer.zero_grad()
            loss = model.forward_GAE(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)

            res = evaluate_embedding(emb, y)
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            accuracies['linearsvc'].append(res[2])
            accuracies['randomforest'].append(res[3])
            print(accuracies['logreg'][-1])

    with torch.no_grad():
        model.eval()
        x_g, emb, y = model.encoder.get_embeddings_v(dataloader_v)
        print(x_g.shape, emb.shape, y.shape)

    np.save('emb/x_g.npy', x_g)
    np.save('emb/x.npy', emb)
    np.save('emb/edge_idx.npy', y)

    assert False

    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('new_log', 'a+') as f:
        accuracies = {'logreg': accuracies['logreg'][-1]}
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
