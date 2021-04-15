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

from rl_common import GraphEdgeEnv, local_args, load_graphs, test_graphs, load_base_model, attackable, get_supervision
from nstep_replay_mem import NstepReplayMem

sys.path.append('%s/../graph_classification' % os.path.dirname(os.path.realpath(__file__)))
from graph_common import loop_dataset

class Agent(object):
    def __init__(self, g_list, test_g_list, env):
        self.g_list = g_list
        if test_g_list is None:
            self.test_g_list = g_list
        else:
            self.test_g_list = test_g_list
        self.mem_pool = NstepReplayMem(memory_size=50000, n_steps=2)
        self.env = env
        # self.net = QNet()
        self.net = NStepQNet(2)
        self.old_net = NStepQNet(2)
        if cmd_args.ctx == 'gpu':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        self.eps_start = 1.0
        self.eps_end = 1.0
        self.eps_step = 10000
        self.burn_in = 100        
        self.step = 0

        self.best_eval = None
        self.pos = 0
        self.sample_idxes = list(range(len(g_list)))
        random.shuffle(self.sample_idxes)
        self.take_snapshot()

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, time_t, greedy=False):
        self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                * (self.eps_step - max(0., self.step)) / self.eps_step)

        if random.random() < self.eps and not greedy:
            actions = self.env.uniformRandActions()
        else:
            cur_state = self.env.getStateRef()
            actions, _, _ = self.net(time_t, cur_state, None, greedy_acts=True)
            actions = list(actions.cpu().numpy())
            
        return actions

    def run_simulation(self):
        if (self.pos + 1) * cmd_args.batch_size > len(self.sample_idxes):
            self.pos = 0
            random.shuffle(self.sample_idxes)

        selected_idx = self.sample_idxes[self.pos * cmd_args.batch_size : (self.pos + 1) * cmd_args.batch_size]
        self.pos += 1
        self.env.setup([self.g_list[idx] for idx in selected_idx])

        t = 0
        while not env.isTerminal():
            list_at = self.make_actions(t)
            list_st = self.env.cloneState()
            self.env.step(list_at)

            assert (env.rewards is not None) == env.isTerminal()
            if env.isTerminal():
                rewards = env.rewards
                s_prime = None
            else:
                rewards = np.zeros(len(list_at), dtype=np.float32)
                s_prime = self.env.cloneState()

            self.mem_pool.add_list(list_st, list_at, rewards, s_prime, [env.isTerminal()] * len(list_at), t)
            t += 1

    def eval(self):
        self.env.setup(deepcopy(self.test_g_list))
        t = 0
        while not self.env.isTerminal():
            list_at = self.make_actions(t, greedy=True)
            self.env.step(list_at)
            t += 1
        test_loss = loop_dataset(env.g_list, env.classifier, list(range(len(env.g_list))), epoch=101)
        print('\033[93m average test: loss %.5f acc %.5f\033[0m' % (test_loss[0], test_loss[1]))

        if cmd_args.phase == 'train' and self.best_eval is None or test_loss[1] < self.best_eval:
            print('----saving to best attacker since this is the best attack rate so far.----')
            torch.save(self.net.state_dict(), cmd_args.save_dir + '/epoch-best.model')
            with open(cmd_args.save_dir + '/epoch-best.txt', 'w') as f:
                f.write('%.4f\n' % test_loss[1])
            self.best_eval = test_loss[1]

        reward = np.mean(self.env.rewards)
        print(reward)
        return reward, test_loss[1]

    def train(self):
        log_out = open(cmd_args.logfile, 'w', 0)
        pbar = tqdm(range(self.burn_in), unit='batch')
        for p in pbar:
            self.run_simulation()
        pbar = tqdm(range(local_args.num_steps), unit='steps')
        optimizer = optim.Adam(self.net.parameters(), lr=cmd_args.learning_rate)
        for self.step in pbar:

            self.run_simulation()

            if self.step % 100 == 0:
                self.take_snapshot()
            if self.step % 100 == 0:
                r, acc = self.eval()
                log_out.write('%d %.6f %.6f\n' % (self.step, r, acc))

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(batch_size=cmd_args.batch_size)

            list_target = torch.Tensor(list_rt)
            if cmd_args.ctx == 'gpu':
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                _, _, banned = zip(*cleaned_sp)
                _, q_t_plus_1, prefix_sum_prime = self.old_net(cur_time + 1, cleaned_sp, None)
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
                list_target[nonterms] = q_rhs
            
            # list_target = get_supervision(self.env.classifier, list_st, list_at)
            list_target = Variable(list_target.view(-1, 1))

            _, q_sa, _ = self.net(cur_time, list_st, list_at)

            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss) )

        log_out.close()
if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    label_map, _, g_list = load_graphs()
    random.shuffle(g_list)
    base_classifier = load_base_model(label_map, g_list)
    env = GraphEdgeEnv(base_classifier, n_edges = 1)
    
    if cmd_args.frac_meta > 0:
        num_train = int( len(g_list) * (1 - cmd_args.frac_meta) )
        agent = Agent(g_list[:num_train], g_list[num_train:], env)
    else:
        agent = Agent(g_list, None, env)

   
    agent.train()

    agent.net.load_state_dict(torch.load(cmd_args.save_dir + '/epoch-best.model'))
    agent.eval()

