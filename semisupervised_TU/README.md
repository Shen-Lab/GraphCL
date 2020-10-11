## Dependencies

Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).
Then, you need to create two directories for pre-trained models and finetuned results to avoid errors:

```
cd ./pre-training
mkdir models
cd ..
cd ./funetuning
mkdir logs
cd ..
```

## Exploring the Role of Data Augmentation in GraphCL

Reproducing the results in the [figure](https://github.com/Shen-Lab/GraphCL/blob/master/augmentations.png) through executing thw followings:

### 1. Pre-training: ###

```
cd ./pre-training
./run_all.sh $DATASET_NAME 0 $GPU_ID
./run_all.sh $DATASET_NAME 5 $GPU_ID
./run_all.sh $DATASET_NAME 10 $GPU_ID
./run_all.sh $DATASET_NAME 15 $GPU_ID
./run_all.sh $DATASET_NAME 20 $GPU_ID
```

### 2. Finetuning: ###

```
cd ./funetuning
./run_all.sh $DATASET_NAME 0 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 5 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 10 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 15 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 20 $EVALUATION_FOLD $GPU_ID
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$EVALUATION_FOLD``` can be 10 or 100 for k-fold evaluation, and ```$GPU_ID``` is the lanched GPU ID.

The scripts will run 5\*5=25 experiments (e.g. ```./run_all.sh $DATASET_NAME 0 $GPU_ID``` will run 5 experiments marked with numbers 0~4 sequentially) and the final results will be recorded in ```./funetuning/logs```
