## Dependencies & Dataset

Please refer to https://github.com/Hanjun-Dai/graph_adversarial_attack for environment setup and to download dataset.
After the configuration, you should have three directories: ```./code/```, ```./dropbox/``` and ```./pytorch_structure2vec/```.

## Training & Evaluation
### Pre-training + finetuning: ###
```
cd ./code/graph_classification
./run_er_components.sh 15 20 0.15 2 -phase train
./run_er_components.sh 15 20 0.15 3 -phase train
./run_er_components.sh 15 20 0.15 4 -phase train

./run_er_components.sh 40 50 0.05 2 -phase train
./run_er_components.sh 40 50 0.05 3 -phase train
./run_er_components.sh 40 50 0.05 4 -phase train

./run_er_components.sh 90 100 0.02 2 -phase train
./run_er_components.sh 90 100 0.02 3 -phase train
./run_er_components.sh 90 100 0.02 4 -phase train
```

### Adversarial attacks: ###
```
cd ./code/graph_attack
./run_trivial.sh 15 20 0.15 2 -phase train
./run_trivial.sh 15 20 0.15 3 -phase train
./run_trivial.sh 15 20 0.15 4 -phase train
./run_grad.sh 15 20 0.15 2 -phase train
./run_grad.sh 15 20 0.15 3 -phase train
./run_grad.sh 15 20 0.15 4 -phase train
./run_dqn.sh 15 20 0.15 2 -phase train
./run_dqn.sh 15 20 0.15 3 -phase train
./run_dqn.sh 15 20 0.15 4 -phase train

./run_trivial.sh 40 50 0.05 2 -phase train
./run_trivial.sh 40 50 0.05 3 -phase train
./run_trivial.sh 40 50 0.05 4 -phase train
./run_grad.sh 40 50 0.05 2 -phase train
./run_grad.sh 40 50 0.05 3 -phase train
./run_grad.sh 40 50 0.05 4 -phase train
./run_dqn.sh 40 50 0.05 2 -phase train
./run_dqn.sh 40 50 0.05 3 -phase train
./run_dqn.sh 40 50 0.05 4 -phase train

./run_trivial.sh 90 100 0.02 2 -phase train
./run_trivial.sh 90 100 0.02 3 -phase train
./run_trivial.sh 90 100 0.02 4 -phase train
./run_grad.sh 90 100 0.02 2 -phase train
./run_grad.sh 90 100 0.02 3 -phase train
./run_grad.sh 90 100 0.02 4 -phase train
./run_dqn.sh 90 100 0.02 2 -phase train
./run_dqn.sh 90 100 0.02 3 -phase train
./run_dqn.sh 90 100 0.02 4 -phase train
```

## Acknowledgements

The backbone implementation is reference to https://github.com/Hanjun-Dai/graph_adversarial_attack.

