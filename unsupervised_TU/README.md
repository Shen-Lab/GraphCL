## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.6.0

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
```

## Training & Evaluation

```
./go.sh $GPU_ID $DATASET_NAME $AUGMENTATION
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.

## Acknowledgements

The backbone implementation is reference to https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.
