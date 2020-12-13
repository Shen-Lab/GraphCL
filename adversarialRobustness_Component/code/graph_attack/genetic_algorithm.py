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

sys.path.append('%s/../common' % os.path.dirname(os.path.realpath(__file__)))
from cmd_args import cmd_args
from graph_embedding import S2VGraph

from rl_common import local_args, load_graphs, load_base_model, attackable

class GeneticAgent(object):
    def __init__(self, classifier, s2v_g, n_edges_attack):
        self.s2v_g = s2v_g
        self.n_edges_attack = n_edges_attack
        self.classifier = classifier
        g = s2v_g.to_networkx()
        comps = [c for c in nx.connected_component_subgraphs(g)]
        self.comps = comps
        self.set_id = {}
        self.solution = None
        for i in range(len(comps)):
            for j in comps[i].nodes():
                self.set_id[j] = i

        self.population = []
        for k in range(cmd_args.population_size):
            added = []
            for k in range(n_edges_attack):
                while True:
                    i = np.random.randint(len(g))
                    j = np.random.randint(len(g))
                    if self.set_id[i] != self.set_id[j] or i == j or (i, j) in added:
                        continue
                    break
                added.append((i, j))
            self.population.append(added)

    def rand_action(self, i):
        region = self.comps[self.set_id[i]].nodes()
        assert len(region) > 1
        while True:
            j = region[np.random.randint(len(region))]
            if j == i:
                continue
            assert self.set_id[i] == self.set_id[j]
            break
        return j

    def get_fitness(self):
        g_list = []
        g = self.s2v_g.to_networkx()
        for edges in self.population:
            g2 = g.copy()
            g2.add_edge(edges[0][0], edges[0][1])
    #        g2.add_edges_from(edges)
            assert nx.number_connected_components(g2) == self.s2v_g.label
            g_list.append(S2VGraph(g2, self.s2v_g.label))

        log_ll, _, acc = self.classifier(g_list)
        acc = acc.cpu().double().numpy()
        if self.solution is None:
            for i in range(len(self.population)):
                if acc[i] < 1.0:
                    self.solution = self.population[i]
                    break
        nll = -log_ll[:, self.classifier.label_map[self.s2v_g.label]]
        return nll

    def select(self, fitness):
        scores = torch.exp(fitness).cpu().data.numpy()
        max_args = np.argsort(-scores)

        result = []
        for i in range(cmd_args.population_size - cmd_args.population_size // 2):
            result.append(deepcopy(self.population[max_args[i]]))

        idx = np.random.choice(np.arange(cmd_args.population_size), 
                                size=cmd_args.population_size // 2,
                                replace=True, 
                                p=scores/scores.sum())
        for i in idx:
            result.append(deepcopy(self.population[i]))                                

        return result

    def crossover(self, parent, pop):
        if np.random.rand() < cmd_args.cross_rate:
            another = pop[ np.random.randint(len(pop)) ]
            if len(parent) != self.n_edges_attack: 
                return another[:]
            if len(another) != self.n_edges_attack:
                return parent[:]
            t = []
            for i in range(self.n_edges_attack):
                if np.random.rand() < 0.5:
                    t.append(parent[i])
                else:
                    t.append(another[i])
            return t
        else:
            return parent[:]

    def mutate(self, child):
        if len(child) != self.n_edges_attack:
            return child
        for i in range(self.n_edges_attack):
            if np.random.rand() < cmd_args.mutate_rate:
                e = child[i]
                if np.random.rand() < 0.5:
                    e = (e[0], self.rand_action(e[0]))
                else:
                    e = (self.rand_action(e[1]), e[1])
                child[i] = e
        return child

    def evolve(self):
        fitness = self.get_fitness()
        if self.solution is not None:
            return
        pop = self.select(fitness)
        new_pop_list = []
        for parent in pop:
            child = self.crossover(parent, pop)
            child = self.mutate(child)
            new_pop_list.append(child)

        self.population = new_pop_list

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    label_map, _, g_list = load_graphs()
    base_classifier = load_base_model(label_map, g_list)

    if cmd_args.idx_start + cmd_args.num_instances > len(g_list):
        instances = g_list[cmd_args.idx_start : ]
    else:
        instances = g_list[cmd_args.idx_start : cmd_args.idx_start + cmd_args.num_instances]

    attacked = 0.0
    pbar = tqdm(instances)
    idx = cmd_args.idx_start
    for g in pbar:
        agent = GeneticAgent(base_classifier, g, cmd_args.num_mod)
        if len(agent.population) == 0:
            continue
        for i in range(cmd_args.rounds):
            agent.evolve()
            if agent.solution is not None:
                attacked += 1
                break
        with open('%s/sol-%d.txt' % (cmd_args.save_dir, idx), 'w') as f:
            f.write('%d: [' % idx)
            if agent.solution is not None:
                for e in agent.solution:
                    f.write('(%d, %d)' % e)
            f.write('] succ: ')
            if agent.solution is not None:
                f.write('1\n')
            else:
                f.write('0\n')
        pbar.set_description('cur_attack: %.2f' % (attacked) )
        idx += 1
    print('\n\nacc: %.4f\n' % ((len(instances) - attacked) / float(len(instances))) )
