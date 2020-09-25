# When Does Self-Supervision Help Graph Convolutional Networks?

PyTorch implementation for [When Does Self-Supervision Help Graph Convolutional Networks?](https://arxiv.org/abs/2006.09136) [[supplement]](https://yyou1996.github.io/files/icml2020_ssgcn_supplement.pdf) [[slides]](https://yyou1996.github.io/files/icml2020_ssgcn_slides.pdf)

Yuning You<sup>\*</sup>, Tianlong Chen<sup>\*</sup>, Zhangyang Wang, Yang Shen

In ICML 2020.

## Overview

Properly designed multi-task self-supervision benefits GCNs in gaining more generalizability and robustness.
In this repository we verify it through performing experiments on several GCN architectures with three designed self-supervised tasks: node clustering, graph partitioning and graph completion.

![](./ssgcn.png)

## Dependencies

Please setup the environment following Section 3 (Setup Python environment for GPU) in this [instruction](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md#3-setup-python-environment-for-gpu), and then install the dependencies related to graph partitioning with the following commands:

```
sudo apt-get install libmetis-dev
pip install METIS==0.2a.4
```

## Experiments

### 1. Pre-training. ###

```
cd ./pre-training
./run_all.sh $DATASET_NAME 0 $GPU_ID
./run_all.sh $DATASET_NAME 5 $GPU_ID
./run_all.sh $DATASET_NAME 10 $GPU_ID
./run_all.sh $DATASET_NAME 15 $GPU_ID
./run_all.sh $DATASET_NAME 20 $GPU_ID
```

### 2. Finetuning. ###

```
cd ./funetuning
./run_all.sh $DATASET_NAME 0 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 5 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 10 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 15 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 20 $EVALUATION_FOLD $GPU_ID
```

## Citation

If you use this code for you research, please cite our paper.

```
@article{you2020does,
  title={When Does Self-Supervision Help Graph Convolutional Networks?},
  author={You, Yuning and Chen, Tianlong and Wang, Zhangyang and Shen, Yang},
  journal={arXiv preprint arXiv:2006.09136},
  year={2020}
}
```

