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

### Pre-training: ###

```
cd ./pre-training
./run_all.sh $DATASET_NAME 0 $GPU_ID
./run_all.sh $DATASET_NAME 5 $GPU_ID
./run_all.sh $DATASET_NAME 10 $GPU_ID
./run_all.sh $DATASET_NAME 15 $GPU_ID
./run_all.sh $DATASET_NAME 20 $GPU_ID
```

### Finetuning: ###

```
cd ./funetuning
./run_all.sh $DATASET_NAME 0 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 5 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 10 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 15 $EVALUATION_FOLD $GPU_ID
./run_all.sh $DATASET_NAME 20 $EVALUATION_FOLD $GPU_ID
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$EVALUATION_FOLD``` can be 10 or 100 for k-fold evaluation, and ```$GPU_ID``` is the lanched GPU ID.

The scripts will run 5\*5=25 experiments (e.g. ```./run_all.sh $DATASET_NAME 0 $GPU_ID``` will run 5 experiments marked with numbers 0~4 sequentially) and the final results will be recorded in ```./funetuning/logs```.

## GraphCL with Sampled Augmentations

Take NCI1 as an example:

### Pre-training: ###

```
cd ./pre-training
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 1
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 3
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 random2 --aug2 random2 --lr 0.001 --suffix 4
```

### Finetuning: ###

```
cd ./funetuning
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 0
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 1
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 2
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 3
CUDA_VISIBLE_DEVICES=$GPU_ID python main_cl.py --dataset NCI1 --aug1 random2 --aug2 random2 --semi_split 100 --model_epoch 100 --suffix 4
```

Five suffixes stand for five runs (with mean & std reported), and augmentations could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.

```lr``` in pre-training should be tuned from {0.01, 0.001, 0.0001} and ```model_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) {20, 40, 60, 80, 100}.

