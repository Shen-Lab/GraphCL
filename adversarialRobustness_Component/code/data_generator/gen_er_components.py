import os
import sys
import cPickle as cp
import random
import numpy as np
import networkx as nx
import time
from tqdm import tqdm


def get_component():
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    g = nx.erdos_renyi_graph(n = cur_n, p = p)

    comps = [c for c in nx.connected_component_subgraphs(g)]
    random.shuffle(comps)
    for i in range(1, len(comps)):
        x = random.choice(comps[i - 1].nodes())
        y = random.choice(comps[i].nodes())
        g.add_edge(x, y)
    assert nx.is_connected(g)
    return g

if __name__ == '__main__':
    save_dir = None
    max_n = None
    min_n = None
    num_graph = None
    p = None
    n_comp = None
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-save_dir':
            save_dir = sys.argv[i + 1]
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-num_graph':
            num_graph = int(sys.argv[i + 1])
        if sys.argv[i] == '-p':
            p = float(sys.argv[i + 1])
        if sys.argv[i] == '-n_comp':
            n_comp = int(sys.argv[i + 1])

    assert save_dir is not None
    assert max_n is not None
    assert min_n is not None
    assert num_graph is not None
    assert p is not None
    assert n_comp is not None

    fout_name = '%s/ncomp-%d-nrange-%d-%d-n_graph-%d-p-%.2f.pkl' % (save_dir, n_comp, min_n, max_n, num_graph, p)
    print('Final Output: ' + fout_name)
    print("Generating graphs...")
    min_n = min_n // n_comp
    max_n = max_n // n_comp

    for i in tqdm(range(num_graph)):

        for j in range(n_comp):
            g = get_component()
            
            if j == 0:
                g_all = g
            else:
                g_all = nx.disjoint_union(g_all, g)
        assert nx.number_connected_components(g_all) == n_comp

        with open(fout_name, 'ab') as fout:
            cp.dump(g_all, fout, cp.HIGHEST_PROTOCOL)
