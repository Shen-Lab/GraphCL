#!/bin/bash -ex
# 0 - 24

aug=("none" "dropN" "permE" "maskN" "subgraph")
a1=$(( $2 % 5 ))
a2=$(( ($2 - $a1) / 5 ))

CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.2 --aug2 ${aug[$a2]} --aug_ratio2 0.2 --suffix 0
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.2 --aug2 ${aug[$a2]} --aug_ratio2 0.2 --suffix 1
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.2 --aug2 ${aug[$a2]} --aug_ratio2 0.2 --suffix 2
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.2 --aug2 ${aug[$a2]} --aug_ratio2 0.2 --suffix 3
CUDA_VISIBLE_DEVICES=$3 python main.py --dataset $1 --aug1 ${aug[$a1]} --aug_ratio1 0.2 --aug2 ${aug[$a2]} --aug_ratio2 0.2 --suffix 4
