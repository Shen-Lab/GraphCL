# Graph Contrastive Learning with Augmentations

PyTorch implementation for [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902) [[supplement]](https://yyou1996.github.io/files/neurips2020_graphcl_supplement.pdf) [[slides]](https://yyou1996.github.io/files/neurips2020_graphcl_slides.pdf)

Yuning You<sup>\*</sup>, Tianlong Chen<sup>\*</sup>, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen

In NeurIPS 2020.

## Overview

In this repository, we develop contrastive learning with augmentations for GNN pre-training (GraphCL, Figure 1) to address the challenge of data heterogeneity in graphs.
Systematic study is performed as shown in Figure 2, to assess the performance of contrasting different augmentations on various types of datasets.

![](./graphcl.png)



![](./augmentations.png)

## Experiments

* [The Role of Data Augmentation](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU#exploring-the-role-of-data-augmentation-in-graphcl)

* Semi-supervised learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU#graphcl-with-sampled-augmentations) [[MNIST and CIFAR10]](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_MNIST_CIFAR10)

* Unsupervised representation learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_TU) [[Cora and Citeseer]](https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_Cora_Citeseer)

## Citation

If you use this code for you research, please cite our paper.

```
@article{you2020graph,
  title={Graph Contrastive Learning with Augmentations},
  author={You, Yuning and Chen, Tianlong and Sui, Yongduo and Chen, Ting and Wang, Zhangyang and Shen, Yang},
  journal={arXiv preprint arXiv:2010.13902},
  year={2020}
}
```

