## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

If you cannot manage to install the old torch-geometric version, one alternative way is to use the new one (maybe ==1.6.0) and make some modifications based on this issue https://github.com/snap-stanford/pretrain-gnns/issues/14.
This might leads to some inconsistent results with those in the paper.

## Training & Evaluation
### Pre-training: ###
```
cd ./bio
python pretrain_graphcl.py --aug1 random --aug2 random
cd ./chem
python pretrain_graphcl.py --aug1 random --aug2 random
```

### Finetuning: ###
```
cd ./bio
./finetune.sh
cd ./chem
./run.sh
```
Results will be recorded in ```result.log```.


## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/pretrain-gnns.
