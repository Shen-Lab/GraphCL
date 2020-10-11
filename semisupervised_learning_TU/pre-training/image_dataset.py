import math
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch_geometric.data import InMemoryDataset, Data


class ImageDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 coord=False,
                 processed_file_prefix='data'):
        assert name in ['MNIST', 'CIFAR10'], "Unsupported data name %s" % name
        self.name = name
        self.coord = coord
        self.processed_file_prefix = processed_file_prefix
        self.traindata = None
        self.testdata = None
        super(ImageDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        if self.name == 'MNIST':
            return ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
                    'train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
        elif self.name == 'CIFAR10':
            return ['data_batch_1', 'data_batch_2', 'data_batch_3',
                    'data_batch_4', 'data_batch_5', 'test_batch']

    @property
    def processed_file_names(self):
        return ['%s_training.pt' % self.processed_file_prefix,
                '%s_test.pt' % self.processed_file_prefix]

    def download(self):
        transform = transforms.ToTensor()
        if self.name == 'CIFAR10':
            data_train = datasets.CIFAR10(root=self.raw_dir,
                                          transform=transform,
                                          train=True,
                                          download=True)
            data_test = datasets.CIFAR10(root=self.raw_dir,
                                         transform=transform,
                                         train=False,
                                         download=True)
        elif self.name == 'MNIST':
            data_train = datasets.MNIST(root=self.raw_dir,
                                        transform=transform,
                                        train=True,
                                        download=True)
            data_test = datasets.MNIST(root=self.raw_dir,
                                       transform=transform,
                                       train=False,
                                       download=True)
        else:
            raise ValueError("Unknown data name {}".format(self.name))
        self.traindata = data_train
        self.testdata = data_test

    def process(self):
        trainLoader = torch.utils.data.DataLoader(self.traindata)
        testLoader = torch.utils.data.DataLoader(self.testdata)
        if self.name == 'MNIST':
            num_row, num_col = 28, 28
        elif self.name == 'CIFAR10':
            num_row, num_col = 32, 32
        else:
            raise ValueError('dataset error')
        num_edges = (3 * num_row - 2) * (3 * num_col - 2)
        edge_index_array = np.zeros(shape=[2, num_edges])
        edge_attr_array = np.zeros(shape=[1, num_edges])
        curt = 0
        for j in range(num_row):
            for k in range(num_col):
                for m in range(max(j-1, 0), min(j+1, num_row-1)+1):
                    for n in range(max(k-1, 0), min(k+1, num_col-1)+1):
                        edge_index_array[0][curt] = j * num_row + k
                        edge_index_array[1][curt] = m * num_row + n
                        edge_attr_array[0][curt] = self.weight(j, k, m, n)
                        curt += 1
        edge_index = torch.from_numpy(edge_index_array).to(torch.int64)
        edge_attr = torch.from_numpy(edge_attr_array).to(torch.float)

        def transform_data(data_loader, edge_index, edge_attr):
            data_list = []
            channel, num_row, num_col = data_loader.dataset[0][0].size()
            if self.coord:
                x = torch.arange(num_col, dtype=torch.float)
                x = x.view((1, -1)).repeat(num_row, 1).view((-1, 1)) - x.mean()
                y = torch.arange(num_row, dtype=torch.float)
                y = y.view((-1, 1)).repeat(1, num_col).view((-1, 1)) - y.mean()
                coord = torch.cat([x, y], -1)

            for image, label in iter(data_loader):
                x = image[0].permute([1,2,0]).view(
                    num_row * num_col, image[0].size()[0])
                if self.coord:
                    x = torch.cat([x, coord], -1)
                data = Data(
                    edge_index=edge_index, edge_attr=edge_attr, x=x, y=label)
                if self.pre_filter is not None:
                    data = self.pre_filter(data)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            return data_list

        train_data_list = transform_data(trainLoader, edge_index, edge_attr)
        torch.save(self.collate(train_data_list), self.processed_paths[0])

        test_data_list = transform_data(testLoader, edge_index, edge_attr)
        torch.save(self.collate(test_data_list), self.processed_paths[1])

    @staticmethod
    def weight(pos_x, pos_y, pos_x_new, pos_y_new):
        dist = (pos_x - pos_x_new) ** 2 + (pos_y - pos_y_new) ** 2
        return math.exp(-dist)

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

