import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.custom_mod import JaggedLogSoftmaxModule, JaggedArgmaxModule, JaggedMaxModule
import sys

def cpu_test():
    mod = JaggedLogSoftmaxModule()
    for i in range(10):
        a = torch.rand(10000, 10)
        b = torch.from_numpy(np.array([ (i + 1) * int(a.size()[1]) for i in range(a.size()[0])]))
        c = mod(Variable(a), Variable(b))
        c2 = F.log_softmax(Variable(a), dim=1)
        print(torch.sum(torch.abs(c - c2)))

    a = torch.rand(100, 30)
    b = torch.from_numpy(np.array([ (i + 1) * 30 for i in range(100)]))
    va = Variable(a, requires_grad=True)
    vb = Variable(b)
    c = mod(va, vb)
    t = F.torch.mean(c)
    t.backward()
    b1 = va.grad

    va = Variable(a, requires_grad=True)
    c = F.log_softmax(va, dim=1)
    t = F.torch.mean(c)
    t.backward()
    b2 = va.grad

    print(torch.sum(torch.abs(b1 - b2)))

def gpu_test():
    mod = JaggedLogSoftmaxModule()
    for i in range(10):
        a = torch.rand(10000, 10).cuda()
        b = torch.from_numpy(np.array([ (i + 1) * int(a.size()[1]) for i in range(a.size()[0])])).cuda()
        c1 = mod(Variable(a), Variable(b))
        c2 = F.log_softmax(Variable(a), dim=1)
        c3 = F.log_softmax(Variable(a.cpu()), dim=1).cuda()
        print(torch.sum(torch.abs(c3 - c2)).data[0], torch.sum(torch.abs(c3 - c1)).data[0], torch.sum(torch.abs(c2 - c1)).data[0])

    a = torch.rand(1000, 100).cuda()
    b = torch.from_numpy(np.array([ (i + 1) * int(a.size()[1]) for i in range(a.size()[0])])).cuda()
    va = Variable(a, requires_grad=True)
    vb = Variable(b)
    c = mod(va, vb)
    t = F.torch.sum(c)
    t.backward()
    b1 = va.grad

    va = Variable(a, requires_grad=True)
    c = F.log_softmax(va, dim=1)
    t = F.torch.sum(c)
    t.backward()
    b2 = va.grad

    va = Variable(a.cpu(), requires_grad=True)
    c = F.log_softmax(va, dim=1)
    t = F.torch.sum(c)
    t.backward()
    b3 = va.grad.cuda()
    print(torch.sum(torch.abs(b3 - b2)).data[0], torch.sum(torch.abs(b3 - b1)).data[0], torch.sum(torch.abs(b2 - b1)).data[0])

def argmax():
    torch.manual_seed(1)    
    mod = JaggedArgmaxModule()

    a = torch.rand(10, 4).cuda()
    print(a)
    b = torch.from_numpy(np.array([ (i + 1) * int(a.size()[1]) for i in range(a.size()[0])])).cuda()
    c = mod(Variable(a), Variable(b))
    print(c)

    a = torch.randn(10).cuda()
    print(a)
    b = torch.LongTensor([2, 5, 9, 10]).cuda()
    c = mod(Variable(a), Variable(b))
    print(c)

torch.manual_seed(1)    
mod = JaggedMaxModule()

a = torch.rand(10, 4).cuda()
print(a)
b = torch.from_numpy(np.array([ (i + 1) * int(a.size()[1]) for i in range(a.size()[0])])).cuda()
c1, c2 = mod(Variable(a), Variable(b))
print(c1)
print(c2)

a = torch.randn(10).cuda()
print(a)
b = torch.LongTensor([2, 5, 9, 10]).cuda()
c = mod(Variable(a), Variable(b))
print(c[0], c[1])