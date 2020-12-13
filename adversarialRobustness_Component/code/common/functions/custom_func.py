import torch
from torch.autograd import Function
from _ext import my_lib
import sys

class JaggedLogSoftmax(Function):
    def forward(self, logits, prefix_sum):        
        self.save_for_backward(prefix_sum)

        assert len(prefix_sum.size()) == 1
        output = logits.new()
        if not logits.is_cuda:
            my_lib.jagged_log_softmax_forward(logits, prefix_sum, output)
        else:
            my_lib.jagged_log_softmax_forward_cuda(logits, prefix_sum, output)

        self.save_for_backward(prefix_sum, output)
        return output

    def backward(self, grad_output):
        prefix_sum, output = self.saved_variables
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            my_lib.jagged_log_softmax_backward(output.data, grad_output, prefix_sum.data, grad_input)
        else:            
            my_lib.jagged_log_softmax_backward_cuda(output.data, grad_output, prefix_sum.data, grad_input)
        return grad_input, None

class JaggedArgmax(Function):
    def forward(self, values, prefix_sum):
        assert len(prefix_sum.size()) == 1
        output = prefix_sum.new()
        if not values.is_cuda:
            my_lib.jagged_argmax_forward(values, prefix_sum, output)
        else:
            my_lib.jagged_argmax_forward_cuda(values, prefix_sum, output)

        return output

    def backward(self, grad_output):
        assert False

class JaggedMax(Function):
    def forward(self, values, prefix_sum):
        assert len(prefix_sum.size()) == 1
        idxes = prefix_sum.new()
        vmax = values.new()
        if not values.is_cuda:
            my_lib.jagged_max_forward(values, prefix_sum, vmax, idxes)
        else:
            my_lib.jagged_max_forward_cuda(values, prefix_sum, vmax, idxes)

        return vmax, idxes

    def backward(self, grad_output):
        assert False

def GraphLaplacianNorm(raw_adj):
    ones = torch.ones(raw_adj.size()[0], 1)
    if raw_adj.is_cuda:
        ones = ones.cuda()
    norm = torch.mm(raw_adj, ones) ** 0.5
    indices = raw_adj._indices()
    values = raw_adj._values()
    if not values.is_cuda:
        my_lib.graph_laplacian_norm(indices, values, norm)
    else:
        my_lib.graph_laplacian_norm_cuda(indices, values, norm)

def GraphDegreeNorm(raw_adj):
    ones = torch.ones(raw_adj.size()[0], 1)
    if raw_adj.is_cuda:
        ones = ones.cuda()
    norm = torch.mm(raw_adj, ones)
    indices = raw_adj._indices()
    values = raw_adj._values()
    if not values.is_cuda:
        my_lib.graph_degree_norm(indices, values, norm)
    else:
        my_lib.graph_degree_norm_cuda(indices, values, norm)
