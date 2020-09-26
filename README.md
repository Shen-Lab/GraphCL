# Pre-Training Graph Neural Networks: A Contrastive Learning Framework with Augmentations

PyTorch implementation for [Pre-Training Graph Neural Networks: A Contrastive Learning Framework with Augmentations]() [[supplement]]() [[slides]]()

Yuning You<sup>\*</sup>, Tianlong Chen<sup>\*</sup>, Yongduo Sui, Ting Chen, Zhangyang Wang, Yang Shen

In NeurIPS 2020.

## Overview

In this repository, we develop contrastive learning with augmentations for GNN pre-training to address the challenge of data heterogeneity in graphs.

![](./graphcl.png)

## Dependencies

Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).

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
TBD
```

