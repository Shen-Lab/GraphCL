# Pre-Training Graph Neural Networks: A Contrastive Learning Framework with Augmentations

PyTorch implementation for [Pre-Training Graph Neural Networks: A Contrastive Learning Framework with Augmentations]() [[supplement]]() [[slides]]()

Yuning You<sup>\*</sup>, Tianlong Chen<sup>\*</sup>, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen

In NeurIPS 2020.

## Overview

In this repository, we develop contrastive learning with augmentations for GNN pre-training (GraphCL) to address the challenge of data heterogeneity in graphs.

![](./graphcl.png)

Systematic study is performed to assess the performance of contrasting different augmentations on various types of datasets, revealing the rationale of the performances and providing the guidance to adopt the framework for specific datasets.

![](./augmentations.png)

## Experiments

* [The Role of Data Augmentation](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU#exploring-the-role-of-data-augmentation-in-graphcl)

* Semi-supervised learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU#graphcl-with-sampled-augmentations) [[MNIST and CIFAR10]](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_MNIST_CIFAR10)

* Unsupervised representation learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_TU) [[Cora and Citeseer]](https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_Cora_Citeseer)

## Citation

If you use this code for you research, please cite our paper.

```
TBD
```

