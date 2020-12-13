## Dependencies & Dataset

Please refer to https://github.com/Hanjun-Dai/graph_adversarial_attack for environment setup and to download dataset.

## Training & Evaluation
### Pre-training + finetuning: ###
```
cd ./code/graph_classification
./run_er_components.sh 15 20 2 -phase train
./run_er_components.sh 15 20 3 -phase train
./run_er_components.sh 15 20 4 -phase train
```

### Adversarial attacks: ###
```
cd ./code/graph_attack
./run_trivial.sh 15 20 2 -phase train
./run_trivial.sh 15 20 3 -phase train
./run_trivial.sh 15 20 4 -phase train
./run_grad.sh 15 20 2 -phase train
./run_grad.sh 15 20 3 -phase train
./run_grad.sh 15 20 4 -phase train
./run_dqn.sh 15 20 2 -phase train
./run_dqn.sh 15 20 3 -phase train
./run_dqn.sh 15 20 4 -phase train
```

## Acknowledgements

The backbone implementation is reference to https://github.com/Hanjun-Dai/graph_adversarial_attack.

