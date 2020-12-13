import cPickle as cp
import networkx as nx

def load_pkl(fname, num_graph):
    g_list = []
    with open(fname, 'rb') as f:
        for i in range(num_graph):
            g = cp.load(f)
            g_list.append(g)
    return g_list

def g2txt(g, label, fid):
    fid.write('%d %d\n' % (len(g), label))
    for i in range(len(g)):
        fid.write('%d' % len(g.neighbors(i)))
        for j in g.neighbors(i):
            fid.write(' %d' % j)
        fid.write('\n')