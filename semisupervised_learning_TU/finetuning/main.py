from functools import partial
from itertools import product

import re
import sys
import argparse
from utils import logger
from datasets import get_dataset
from train_eval import cross_validation_with_val_set, single_train_test


DATA_SOCIAL = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI']
DATA_SOCIAL += ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY']
DATA_BIO = ['MUTAG', 'NCI1', 'PROTEINS', 'DD', 'ENZYMES', 'PTC_MR']
DATA_REDDIT = [
    data for data in DATA_BIO + DATA_SOCIAL if "REDDIT" in data]
DATA_NOREDDIT = [
    data for data in DATA_BIO + DATA_SOCIAL if "REDDIT" not in data]
DATA_SUBSET_STUDY = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                     'NCI1', 'PROTEINS', 'DD']
DATA_SUBSET_STUDY_SUP = [
    d for d in DATA_SOCIAL + DATA_BIO if d not in DATA_SUBSET_STUDY]
DATA_SUBSET_FAST = ['IMDB-BINARY', 'PROTEINS', 'IMDB-MULTI', 'ENZYMES']
DATA_IMAGES = ['MNIST', 'MNIST_SUPERPIXEL', 'CIFAR10']


str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default="benchmark")
parser.add_argument('--data_root', type=str, default="data")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=500)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--model', type=str, default="infomax")
parser.add_argument('--model_lr', type=str, default="0.001")
parser.add_argument('--model_epoch', type=str, default="100")
parser.add_argument('--dataset', type=str, default="NCI1")
parser.add_argument('--semi_split', type=int, default=10)
parser.add_argument('--suffix', type=str, default="0")
args = parser.parse_args()

if args.model == "gae":
    from net_gae import ResGCN
elif args.model == "infomax":
    from net_infomax import ResGCN
elif args.model == "cl":
    from net_cl import ResGCN

if args.model == "gae":
    model_PATH = "../../gfn-" + args.model + "/models/" + args.dataset + "_" + args.model_lr + "_" + args.model_epoch + "_" + args.suffix + ".pt"
elif args.model == "infomax":
    model_PATH = "../../gfn-" + args.model + "/models/" + args.dataset + "_" + args.model_lr + "_" + args.model_epoch + " " + args.suffix + ".pt"

log_PATH = "logs1/" + args.model + "_" + args.dataset + "_log_" + str(args.semi_split) + "fold"
if args.model == "cl":
    model_PATH = "models_cl/" + args.dataset + "_random_0.2_random_0.2_" + args.model_epoch + "_" + args.model_lr + ".pt"

def create_n_filter_triples(datasets, feat_strs, nets, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    triples = [(d, f, n) for d, f, n in product(datasets, feat_strs, nets)]
    triples_filtered = []
    for dataset, feat_str, net in triples:
        # Add ak3 for GFN.
        if gfn_add_ak3 and 'GFN' in net:
            feat_str += '+ak3'
        # Remove edges for GFN.
        if gfn_reall and 'GFN' in net:
            feat_str += '+reall'
        # Replace degree feats for REDDIT datasets (less redundancy, faster).
        if reddit_odeg10 and dataset in [
                'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
        # Replace degree and akx feats for dd (less redundancy, faster).
        if dd_odeg10_ak1 and dataset in ['DD']:
            feat_str = feat_str.replace('odeg100', 'odeg10')
            feat_str = feat_str.replace('ak3', 'ak1')
        triples_filtered.append((dataset, feat_str, net))
    return triples_filtered


def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm

    # modify default architecture when needed
    if model_name.find('_') > 0:
        num_conv_layers_ = re.findall('_conv(\d+)', model_name)
        if len(num_conv_layers_) == 1:
            num_conv_layers = int(num_conv_layers_[0])
            print('[INFO] num_conv_layers set to {} as in {}'.format(
                num_conv_layers, model_name))
        num_fc_layers_ = re.findall('_fc(\d+)', model_name)
        if len(num_fc_layers_) == 1:
            num_fc_layers = int(num_fc_layers_[0])
            print('[INFO] num_fc_layers set to {} as in {}'.format(
                num_fc_layers, model_name))
        residual_ = re.findall('_res(\d+)', model_name)
        if len(residual_) == 1:
            residual = bool(int(residual_[0]))
            print('[INFO] residual set to {} as in {}'.format(
                residual, model_name))
        gating = re.findall('_gating', model_name)
        if len(gating) == 1:
            global_pool += "_gating"
            print('[INFO] add gating to global_pool {} as in {}'.format(
                global_pool, model_name))
        dropout_ = re.findall('_drop([\.\d]+)', model_name)
        if len(dropout_) == 1:
            dropout = float(dropout_[0])
            print('[INFO] dropout set to {} as in {}'.format(
                dropout, model_name))
        hidden_ = re.findall('_dim(\d+)', model_name)
        if len(hidden_) == 1:
            hidden = int(hidden_[0])
            print('[INFO] hidden set to {} as in {}'.format(
                hidden, model_name))



    if model_name.startswith('ResGFN'):
        collapse = True if 'flat' in model_name else False
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=True, collapse=collapse,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    elif model_name.startswith('ResGCN'):
        def foo(dataset):
            return ResGCN(dataset, hidden, num_feat_layers, num_conv_layers,
                          num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)
    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo


def run_exp_lib(dataset_feat_net_triples,
                get_model=get_model_with_default_configs):
    results = []
    exp_nums = len(dataset_feat_net_triples)
    print("-----\nTotal %d experiments in this run:" % exp_nums)
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
    print("Here we go..")
    sys.stdout.flush()
    for exp_id, (dataset_name, feat_str, net) in enumerate(
            dataset_feat_net_triples):
        print('-----\n{}/{} - {} - {} - {}'.format(
            exp_id+1, exp_nums, dataset_name, feat_str, net))
        sys.stdout.flush()
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        model_func = get_model(net)
        if 'MNIST' in dataset_name or 'CIFAR' in dataset_name:
            train_dataset, test_dataset = dataset
            train_acc, acc, duration = single_train_test(
                train_dataset,
                test_dataset,
                model_func,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                epoch_select=args.epoch_select,
                with_eval_mode=args.with_eval_mode)
            std = 0
        else:
            train_acc, acc, std, duration = cross_validation_with_val_set(
                dataset,
                model_func,
                folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                epoch_select=args.epoch_select,
                with_eval_mode=args.with_eval_mode,
                logger=logger, model_PATH=model_PATH, semi_split=args.semi_split)

        with open(log_PATH, "a+") as f:
            f.write(args.model_lr + " " + args.model_epoch + ": ")
            f.write(str(acc) + " " + str(std))
            f.write("\n")

        summary1 = 'data={}, model={}, feat={}, eval={}'.format(
            dataset_name, net, feat_str, args.epoch_select)
        summary2 = 'train_acc={:.2f}, test_acc={:.2f} ± {:.2f}, sec={}'.format(
            train_acc*100, acc*100, std*100, round(duration, 2))
        results += ['{}: {}, {}'.format('fin-result', summary1, summary2)]
        print('{}: {}, {}'.format('mid-result', summary1, summary2))
        sys.stdout.flush()
    print('-----\n{}'.format('\n'.join(results)))
    sys.stdout.flush()


def run_exp_arch_res_n_layers(gfn=False, gcn=False, resnet=False):
    print('[INFO] running architecture ablation on conv depth and resnet..')
    # datasets = DATA_SUBSET_STUDY
    # datasets = DATA_SUBSET_STUDY_SUP
    datasets = DATA_BIO + DATA_SOCIAL
    feat_strs = ['deg+odeg100']
    cf_triples = partial(create_n_filter_triples, gfn_add_ak3=True,
                         reddit_odeg10=True, dd_odeg10_ak1=True)

    # Test num layers for GFN
    if gfn:
        nets = ['ResGFN']
        nets_new = ['ResGFN-flat_fc1']
        for num_fc_layers in [2, 1]:
            for num_conv_layers in [0, 1, 2, 3, 4]:
                for net in nets:
                    net_new = '{}_conv{}_fc{}'.format(
                        net, num_conv_layers, num_fc_layers)
                    nets_new.append(net_new)
        run_exp_lib(cf_triples(datasets, feat_strs, nets_new))

    # Test num layers for GCN
    if gcn:
        nets = ['ResGCN']
        nets_new = []
        for num_conv_layers in [0, 1, 2, 3, 4]:
            for net in nets:
                net_new = '{}_conv{}_fc2'.format(
                    net, num_conv_layers)
                nets_new.append(net_new)
        run_exp_lib(cf_triples(datasets, feat_strs, nets_new))

    # Test residual connection
    if resnet:
        nets = ['ResGFN', 'ResGCN']
        nets_new = []
        for num_conv_layers in [3]:
            for residual in [0, 1]:
                for net in nets:
                    net_new = '{}_conv{}_fc2_res{}'.format(
                        net, num_conv_layers, residual)
                    nets_new.append(net_new)
        run_exp_lib(cf_triples(datasets, feat_strs, nets_new))


def run_exp_feat_study():
    print('[INFO] running feature study..')
    # datasets = DATA_SUBSET_STUDY
    # datasets = DATA_NOREDDIT
    datasets = DATA_BIO + DATA_SOCIAL
    feat_strs = ['none', 'deg+odeg100', 'ak1', 'ak2', 'ak3', 'cent']
    feat_strs += ['deg+odeg100+ak1', 'deg+odeg100+ak2', 'deg+odeg100+ak3']
    feat_strs += ['deg+odeg100+ak3+cent']
    nets = ['ResGFN', 'ResGCN']
    run_exp_lib(create_n_filter_triples(datasets, feat_strs, nets,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=False))


def run_exp_benchmark():
    # Run GFN, GFN (light), GCN
    print('[INFO] running standard benchmarks..')
    # datasets = DATA_BIO + DATA_SOCIAL
    datasets = [args.dataset]
    feat_strs = ['deg+odeg100']
    # nets = ['ResGFN', 'ResGFN_conv0_fc2', 'ResGCN']
    nets = ['ResGCN']
    run_exp_lib(create_n_filter_triples(datasets, feat_strs, nets,
                                        gfn_add_ak3=True,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=True))


def run_exp_noises():
    # Run GFN, GCN
    print('[INFO] running noises experiments..')
    datasets = DATA_BIO + DATA_SOCIAL
    # feat_strs = ['deg+odeg100+randd0.%d'%d for d in range(10)]  # Randomly delete edges
    # feat_strs = ['deg+odeg100+randa%f'%f for f in [0, 0.5, 1.0, 2.0, 5.0, 10.0]]  # Randomly add edges
    feat_strs = ['deg+odeg100+randa%f+randd%f'%(f, f) for f in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]  # Randomly add/delete edges
    nets = ['ResGFN', 'ResGCN']
    run_exp_lib(create_n_filter_triples(datasets, feat_strs, nets,
                                        gfn_add_ak3=True,
                                        reddit_odeg10=True,
                                        dd_odeg10_ak1=True))


def run_exp_image(nets=['ResGCN'], feat_strs=['none'], datasets=['MNIST']):
    # Test num layers for GFN
    nets_new = []
    for num_fc_layers in [2]:
        for num_conv_layers in [3, 5, 7]:
            for net in nets:
                net_new = '{}_conv{}_fc{}'.format(
                    net, num_conv_layers, num_fc_layers)
                nets_new.append(net_new)
    run_exp_lib(create_n_filter_triples(datasets, feat_strs, nets_new))


def run_exp_single_test():
    print('[INFO] running single test..')
    run_exp_lib([('MUTAG', 'deg+odeg100+ak3+reall', 'ResGFN')])
    #run_exp_lib([('IMDB-BINARY', 'none', 'ResGCN')])


if __name__ == '__main__':
    if args.exp == 'test':
        run_exp_single_test()
    elif args.exp == 'benchmark':
        run_exp_benchmark()
    elif args.exp == 'noises':
        run_exp_noises()
    elif args.exp == 'image_gcn':
        run_exp_image(nets=['ResGCN'], feat_strs=['none'])
    elif args.exp == 'image_gfn':
        run_exp_image(nets=['ResGFN'], feat_strs=['ak3', 'ak5', 'ak7'])
    elif args.exp == 'feature_study':
        run_exp_feat_study()
    elif args.exp == 'arc_study_gfn':
        run_exp_arch_res_n_layers(gfn=True)
    elif args.exp == 'arc_study_gcn':
        run_exp_arch_res_n_layers(gcn=True)
    else:
        raise ValueError('Unknown exp {} to run'.format(args.exp))
    pass
