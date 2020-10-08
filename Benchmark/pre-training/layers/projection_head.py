import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class projection_head(nn.Module):

    def __init__(self, input_dim, output_dim): #L=nb_hidden_layers
        super().__init__()
        self.fc_layer1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc_layer2 = nn.Linear(input_dim, input_dim, bias=True)  
        
        
    def forward(self, x):
        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        return x