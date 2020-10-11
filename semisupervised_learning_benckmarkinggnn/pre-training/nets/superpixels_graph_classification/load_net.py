"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.superpixels_graph_classification.gcn_net import GCNNet
from nets.superpixels_graph_classification.gat_net import GATNet
from nets.superpixels_graph_classification.gin_net import GINNet


def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GIN(net_params):
    return GINNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        
        'GCN': GCN,
        'GAT': GAT,
        'GIN': GIN
        
    }
        
    return models[MODEL_NAME](net_params)