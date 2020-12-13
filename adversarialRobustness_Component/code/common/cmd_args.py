import argparse
import cPickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser for molecule vae')
cmd_opt.add_argument('-data_folder', type=str, default=None, help='data folder')
cmd_opt.add_argument('-saved_model', type=str, default=None, help='saved model')
cmd_opt.add_argument('-save_dir', type=str, default=None, help='save folder')
cmd_opt.add_argument('-ctx', type=str, default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-phase', type=str, default='test', help='train/test')
cmd_opt.add_argument('-logfile', type=str, default=None, help='log')

cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-min_n', type=int, default=0, help='min #nodes')
cmd_opt.add_argument('-max_n', type=int, default=0, help='max #nodes')
cmd_opt.add_argument('-min_c', type=int, default=0, help='min #components')
cmd_opt.add_argument('-max_c', type=int, default=0, help='max #components')
cmd_opt.add_argument('-er_p', type=float, default=0, help='parameter of er graphs')
cmd_opt.add_argument('-n_graphs', type=int, default=0, help='number of graphs')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp/gcn')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=0, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=32, help='dimension of classification')
cmd_opt.add_argument('-max_lv', type=int, default=2, help='max rounds of message passing')

cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-learning_rate', type=float, default=0.001, help='init learning_rate')
cmd_opt.add_argument('-weight_decay', type=float, default=5e-4, help='weight_decay')
cmd_opt.add_argument('-dropout', type=float, default=0.5, help='dropout rate')

# for node classification
cmd_opt.add_argument('-dataset', type=str, default=None, help='citeseer/cora/pubmed')
cmd_opt.add_argument('-feature_dim', type=int, default=None, help='node feature dim')
cmd_opt.add_argument('-num_class', type=int, default=None, help='# classes')
cmd_opt.add_argument('-adj_norm', type=int, default=1, help='normalize the adj or not')

# for bio graph classification
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of node feature')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')

# for attack 
cmd_opt.add_argument('-idx_start', type=int, default=None, help='id of graph or node index')
cmd_opt.add_argument('-num_instances', type=int, default=None, help='num of samples for attack, in genetic algorithm')
cmd_opt.add_argument('-num_steps', type=int, default=100000, help='rl training steps')
cmd_opt.add_argument('-targeted', type=int, default=0, help='0/1 target attack or not')
cmd_opt.add_argument('-frac_meta', type=float, default=0, help='fraction for meta rl learning')
cmd_opt.add_argument('-meta_test', type=int, default=0, help='for meta rl learning')
cmd_opt.add_argument('-rand_att_type', type=str, default=None, help='random/exhaust')
cmd_opt.add_argument('-reward_type', type=str, default=None, help='binary/nll')
cmd_opt.add_argument('-base_model_dump', type=str, default=None, help='saved base model')
cmd_opt.add_argument('-num_mod', type=int, default=1, help='number of modifications allowed')

# for genetic algorithm
cmd_opt.add_argument('-population_size', type=int, default=100, help='population size')
cmd_opt.add_argument('-cross_rate', type=float, default=0.1, help='cross_rate')
cmd_opt.add_argument('-mutate_rate', type=float, default=0.2, help='mutate rate')
cmd_opt.add_argument('-rounds', type=int, default=10, help='rounds of evolution')

# for node attack
cmd_opt.add_argument('-bilin_q', type=int, default=0, help='bilinear q or not')
cmd_opt.add_argument('-mlp_hidden', type=int, default=64, help='mlp hidden layer size')
cmd_opt.add_argument('-n_hops', type=int, default=2, help='attack range')

# for defence
cmd_opt.add_argument('-del_rate', type=float, default=0, help='rate of deleting edge')

cmd_args, _ = cmd_opt.parse_known_args()

print(cmd_args)

def build_kwargs(keys, arg_dict):
    st = ''
    for key in keys:
        st += '%s-%s' % (key, str(arg_dict[key]))
    return st

def save_args(fout, args):
    with open(fout, 'wb') as f:
        cp.dump(args, f, cp.HIGHEST_PROTOCOL)
