## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

## Training & Evaluation
### Pre-training: ###
```
cd ./bio
python pretrain_graphcl.py --output_model_file ./models_graphcl
cd ./chem
python pretrain_graphcl.py --output_model_file ./models_graphcl
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
