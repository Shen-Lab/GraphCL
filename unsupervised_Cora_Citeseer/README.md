# GraphCL in Node-Classification Tasks

## 1. Requirements

`` python==3.6.2``

`` pytorch==1.5.0``

## 2. Command example

For subgraph augmentation in contrastive learning on citeseer dataset:

   `python -u execute.py --dataset citeseer --aug_type subgraph --drop_percent 0.20 --seed 39 --save_name cite_best_dgi.pkl --gpu 5`

## Data
For the data description please kindly refer to https://github.com/kimiyoung/planetoid#prepare-the-data.
