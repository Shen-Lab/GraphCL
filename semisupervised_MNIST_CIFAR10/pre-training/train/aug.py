import torch
import torch.nn as nn
import dgl
from random import randint
import random
import copy
import pdb
import numpy as np
from collections import Counter


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
double
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_double(graph_list, type):
    type1 = type[0]
    type2 = type[1]
    if type1 == 'n':
        aug_list1 = aug_drop_node_list(graph_list, 0.2)
    elif type1 == 'l':
        aug_list1 = aug_drop_add_link_list(graph_list, 0.2)
    elif type1 == 'm':
        aug_list1 = aug_mask_list(graph_list, 0.2)
    elif type1 == 's':
        aug_list1 = aug_subgraph_list(graph_list, 0.4)
    elif type1 == 'o':
        aug_list1 = graph_list
        
    if type2 == 'n':
        aug_list2 = aug_drop_node_list(graph_list, 0.2)
    elif type2 == 'l':
        aug_list2 = aug_drop_add_link_list(graph_list, 0.2)
    elif type2 == 'm':
        aug_list2 = aug_mask_list(graph_list, 0.2)
    elif type2 == 's':
        aug_list2 = aug_subgraph_list(graph_list, 0.4)
    elif type2 == 'o':
        aug_list2 = graph_list
    return aug_list1, aug_list2

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
random3
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_random3(graph_list):
    graph_num = len(graph_list)
    aug_list =[]
    random_list = torch.randint(1, 4, (graph_num,)).tolist()
    aug_count = Counter(random_list)
    for i in range(graph_num):
        if random_list[i] == 1:
            aug_graph = aug_drop_node(graph_list[i], drop_percent=0.2)
        elif random_list[i] == 2:
            aug_graph = aug_drop_add_link(graph_list[i], drop_percent=0.2)
        elif random_list[i] == 3:
            aug_graph = aug_subgraph(graph_list[i], drop_percent=0.4)
        aug_list.append(aug_graph)
    return aug_list, aug_count

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
random
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_random(graph_list):
    graph_num = len(graph_list)
    aug_list =[]
    random_list = torch.randint(1, 5, (graph_num,)).tolist()
    aug_count = Counter(random_list)
    for i in range(graph_num):
        if random_list[i] == 1:
            aug_graph = aug_drop_node(graph_list[i], drop_percent=0.2)
        elif random_list[i] == 2:
            aug_graph = aug_drop_add_link(graph_list[i], drop_percent=0.2)
        elif random_list[i] == 3:
            aug_graph = aug_mask(graph_list[i], drop_percent=0.2)
        elif random_list[i] == 4:
            aug_graph = aug_subgraph(graph_list[i], drop_percent=0.4)
        aug_list.append(aug_graph)
    return aug_list, aug_count


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
drop nodes
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def aug_drop_node_list(graph_list, drop_percent):
    
    graph_num = len(graph_list) # number of graphs
    aug_list = []
    for i in range(graph_num):
        aug_graph = aug_drop_node(graph_list[i], drop_percent)
        aug_list.append(aug_graph)
    return aug_list


def aug_drop_node(graph, drop_percent=0.2):

    num = graph.number_of_nodes() # number of nodes of one graph
    drop_num = int(num * drop_percent)    # number of drop nodes
    aug_graph = copy.deepcopy(graph)
    all_node_list = [i for i in range(num)]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph.remove_nodes(drop_node_list)
    return aug_graph

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
drop add links
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def aug_drop_add_link_list(graph_list, drop_percent):
    
    graph_num = len(graph_list) # number of graphs
    aug_list = []
    for i in range(graph_num):
        aug_graph = aug_drop_add_link(graph_list[i], drop_percent)
        aug_list.append(aug_graph)
    return aug_list


def aug_drop_add_link(graph, drop_percent=0.2):

    pro = drop_percent / 2
    aug_graph = copy.deepcopy(graph)
    edge_num = aug_graph.number_of_edges()

    drop_num = int(edge_num * pro / 2) 
    add_num = int(edge_num * pro / 2)  
    del_edges_id_list = [] 
    all_edges_id_list = [i for i in range(edge_num)]
    
    for i in range(drop_num):

        random_idx = randint(0, edge_num - 1) 
        u_v = aug_graph.find_edges(all_edges_id_list[random_idx]) 
        del_edge_id1 = aug_graph.edge_ids(u_v[0], u_v[1])
        del_edge_id2 = aug_graph.edge_ids(u_v[1], u_v[0])
        if del_edge_id1.size(0):
            del_edges_id_list.append(del_edge_id1)
            all_edges_id_list.remove(del_edge_id1.item())
        if del_edge_id2.size(0):
            del_edges_id_list.append(del_edge_id2)
            all_edges_id_list.remove(del_edge_id2.item())
        edge_num -= 2
    aug_graph.remove_edges(del_edges_id_list)
    '''
        above finish drop edges
    '''
    node_num = aug_graph.number_of_nodes() 
    l = [[i, j] for i in range(node_num) for j in range(i)]
    d = torch.tensor(random.sample(l, add_num))
    add_edges_src_list = d.t()[0]
    add_edges_dst_list = d.t()[1]
    aug_graph.add_edges(add_edges_src_list, add_edges_dst_list)
    aug_graph.add_edges(add_edges_dst_list, add_edges_src_list)

    return aug_graph


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
mask
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def aug_mask_list(graph_list, drop_percent):
    
    graph_num = len(graph_list) # number of graphs
    aug_list = []
    for i in range(graph_num):
        aug_graph = aug_mask(graph_list[i], drop_percent)
        aug_list.append(aug_graph)
    return aug_list


def aug_mask(graph, drop_percent=0.2):
    
    num = graph.number_of_nodes() 
    mask_num = int(num * drop_percent) 
    node_idx = [i for i in range(num)]
    mask_list = random.sample(node_idx, mask_num)
    aug_graph = copy.deepcopy(graph)
    zeros = torch.zeros_like(aug_graph.ndata['feat'][0])
    for j in mask_list:
        aug_graph.ndata['feat'][j] = zeros
    return aug_graph


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
subgraph
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_subgraph_list(graph_list, drop_percent):
    
    graph_num = len(graph_list)
    aug_list = []
    for i in range(graph_num):
        s_graph = aug_subgraph(graph_list[i], drop_percent)
        aug_list.append(s_graph)
    return aug_list


def aug_subgraph(graph, drop_percent):

    graph = copy.deepcopy(graph)
    num = graph.number_of_nodes()
    all_node_list = [i for i in range(num)]
    s_num = int(num * (1 -drop_percent))
    center_node_id = random.randint(0, num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []
    for i in range(s_num - 1):
        
        all_neighbor_list += graph.successors(sub_node_id_list[i]).numpy().tolist()
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break
    del_node_list = [i for i in all_node_list if not i in sub_node_id_list]
    graph.remove_nodes(del_node_list)
    return graph


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
new
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_new_list(graph_list, threshold):
    
    graph_num = len(graph_list)
    aug_list = []
    for i in range(graph_num):
        s_graph = aug_new(graph_list[i], threshold)
        aug_list.append(s_graph)
    return aug_list

def aug_new(graph, threshold):

    node_feature_matrix = graph.ndata['h']
    adjacent = torch.mm(node_feature_matrix, node_feature_matrix.t())
    adjacent_s = torch.sigmoid(adjacent)
    add_edge_list = (adjacent_s > threshold).nonzero()
    src = add_edge_list.t()[0]
    dst = add_edge_list.t()[1]
    aug_graph = copy.deepcopy(graph)
    aug_graph.remove_edges([j for j in range(aug_graph.number_of_edges())])
    aug_graph.add_edges(src, dst)
    aug_graph.add_edges(dst, src)
    return aug_graph


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
tsp dataset drop add links
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_tsp_drop_add_link(graph_list, drop_percent=0.2):

    pro = drop_percent / 2 # 0.1    160 * 0.1 = 16 
    graph_num = len(graph_list) 
    aug_list = []
    
    for i in range(graph_num):
        
        aug_graph = copy.deepcopy(graph_list[i])
        edge_num = aug_graph.number_of_edges()
        drop_num = int(edge_num * pro) 
        add_num = int(edge_num * pro)   
        all_edges_id_list = [i for i in range(edge_num)]
        del_edges_id_list = random.sample(all_edges_id_list, drop_num)
        aug_graph.remove_edges(del_edges_id_list)
        '''
            above finish drop edges
        '''
        node_num = aug_graph.number_of_nodes() 
        l = [(i, j) for i in range(node_num) for j in range(node_num)]
        d = random.sample(l, add_num)
        add_edges_src_list = []
        add_edges_dst_list = []
        for i in range(add_num):
            add_edges_src_list.append(d[i][0])
            add_edges_dst_list.append(d[i][1])
        aug_graph.add_edges(add_edges_src_list, add_edges_dst_list)
        aug_list.append(aug_graph)
    return aug_list


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dd dataset subgraph
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_dd_subgraph_list(graph_list, drop_percent):
    
    graph_num = len(graph_list)
    aug_list = []
    for i in range(graph_num):
        s_graph = aug_dd_subgraph(graph_list[i], drop_percent)
        aug_list.append(s_graph)
    return aug_list


def aug_dd_subgraph(ori_graph, drop_percent):
    graph = copy.deepcopy(ori_graph)
    num = graph.number_of_nodes()
    if num > 2000:
        return graph
    else:
        all_node_list = [i for i in range(num)]
        s_num = int(num * (1 -drop_percent))
        center_node_id = random.randint(0, num - 1)

        sub_node_id_list = [center_node_id]
        all_neighbor_list = []
        for i in range(s_num - 1):
            
            all_neighbor_list += graph.successors(sub_node_id_list[i]).numpy().tolist()
            all_neighbor_list = list(set(all_neighbor_list))
            new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]

            if len(new_neighbor_list) != 0:
                new_node = random.sample(new_neighbor_list, 1)[0]
                sub_node_id_list.append(new_node)
            else:
                break
    
        del_node_list = [i for i in all_node_list if not i in sub_node_id_list]
        graph.remove_nodes(del_node_list)
        return graph


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
add links
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_add_edges(graph_list, drop_percent=0.2):

    graph_num = len(graph_list) 
    aug_list = []
    for i in range(graph_num):

        aug_graph = copy.deepcopy(graph_list[i])
        edge_num = aug_graph.number_of_edges()
        add_num = int(edge_num * drop_percent / 2)

        node_num = aug_graph.number_of_nodes()
        l = []
        for i in range(node_num):
            for j in range(i):
                l.append((i, j))
        d = random.sample(l, add_num)

        add_edges_src_list = []
        add_edges_dst_list = []

        for i in range(add_num):
            if not aug_graph.has_edge_between(d[i][0], d[i][1]):
                add_edges_src_list.append(d[i][0])
                add_edges_src_list.append(d[i][1])
                add_edges_dst_list.append(d[i][1])
                add_edges_dst_list.append(d[i][0])
        aug_graph.add_edges(add_edges_src_list, add_edges_dst_list)
        aug_list.append(aug_graph)
    return aug_list


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
drop links
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def aug_drop_edges(graph_list, drop_percent=0.2):

    graph_num = len(graph_list) 
    aug_list = []
    for i in range(graph_num):

        aug_graph = copy.deepcopy(graph_list[i])
        edge_num = aug_graph.number_of_edges()
        drop_num = int(edge_num * drop_percent / 2) 
        
        del_edges_id_list = [] 
        all_edges_id_list = [i for i in range(edge_num)]
        for i in range(drop_num):

            random_idx = randint(0, edge_num - 1) 
            u_v = aug_graph.find_edges(all_edges_id_list[random_idx])
            del_edge_id1 = aug_graph.edge_ids(u_v[0], u_v[1])
            del_edge_id2 = aug_graph.edge_ids(u_v[1], u_v[0])
            del_edges_id_list.append(del_edge_id1)
            del_edges_id_list.append(del_edge_id2)
            all_edges_id_list.remove(del_edge_id1.item())
            all_edges_id_list.remove(del_edge_id2.item())
            edge_num -= 2

        aug_graph.remove_edges(del_edges_id_list)
        aug_list.append(aug_graph)
    return aug_list

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
add noise
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def add_guassian_noise(graph_list, drop_percent=1):

    graph_num = len(graph_list) 
    aug_list = []
    for i in range(graph_num):
        aug_graph = copy.deepcopy(graph_list[i])
        noise = torch.randn(aug_graph.ndata['feat'].shape) * drop_percent
        aug_graph.ndata['feat'] += noise
        aug_list.append(aug_graph)
    return aug_list


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SBM dataset mask
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def add_SBM_mask(graph_list, drop_percent=0.2):
    
    graph_num = len(graph_list) 
    aug_list = []
    for i in range(graph_num):
        num = graph_list[i].number_of_nodes() 
        mask_num = int(num * drop_percent) 
        node_idx = [i for i in range(num)]
        mask_list = random.sample(node_idx, mask_num)
        aug_graph = copy.deepcopy(graph_list[i])
        for j in mask_list:
            aug_graph.ndata['feat'][j] = 3
        aug_list.append(aug_graph)
    return aug_list


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
batched
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def collate_batched_graph(graphs):
        # The input samples is a list of pairs (graph, label).
    tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
    snorm_n = torch.cat(tab_snorm_n).sqrt()  
    tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
    
    while 0 in tab_sizes_e:
        tab_sizes_e[tab_sizes_e.index(0)] = 1
        
    tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
    snorm_e = torch.cat(tab_snorm_e).sqrt()
    batched_graph = dgl.batch(graphs)
    return batched_graph, snorm_n, snorm_e

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
sim matrix
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def sim_matrix2(ori_vector, arg_vector, temp=1.0):
    
    for i in range(len(ori_vector)):
        sim = torch.cosine_similarity(ori_vector[i].unsqueeze(0), arg_vector, dim=1) * (1/temp)
        if i == 0:
            sim_tensor = sim.unsqueeze(0)
        else:
            sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), 0)
    return sim_tensor

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
compute
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def compute_diag_sum(tensor):
    num = len(tensor)
    diag_sum = 0
    for i in range(num):
        diag_sum += tensor[i][i]
    return diag_sum


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
visual a graph
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def vis(g, name):
    import networkx as nx
    import matplotlib.pyplot as plt

    nx.draw(g.to_networkx(), with_labels=True)
    plt.savefig('./'+ str(name) +'.png')
    plt.show()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
compute acc
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def aug_acc(inputs, device, topk=(1, 5)):
    """Computes acc """
    
    maxk = max(topk)
    batch_size = inputs.size(0)
    _, pred = inputs.topk(maxk, 1, True, True)
    pred = pred.t()
    target = torch.arange(batch_size)
    target = target.view(1, -1).expand_as(pred).to(device)
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
AverageMeter
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
vis_a_batch_of_graph_data
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def vis_a_batch_of_graph_data(graph_list):
    num = len(graph_list)
    node_num_list = []
    edge_num_list = []
    for i in range(num):
        node_num_list.append(graph_list[i].number_of_nodes())
        edge_num_list.append(graph_list[i].number_of_edges())
    return node_num_list, edge_num_list
