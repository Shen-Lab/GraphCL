#!/bin/bash

dropbox=../../dropbox

min_n=$1
max_n=$2
p=$3
min_c=1
max_c=3
base_lv=$4
data_folder=$dropbox/data/components
save_fold=nodes-${min_n}-${max_n}-p-${p}-c-${min_c}-${max_c}-lv-${base_lv}
base_model_dump=$dropbox/scratch/results/graph_classification/components/$save_fold/epoch-best

idx_start=0
num=2000
pop=50
cross=0.1
mutate=0.2
rounds=10

output_base=$HOME/scratch/results/graph_classification/components/$save_fold
output_root=$output_base/ga-p-${pop}-c-${cross}-m-${mutate}-r-${rounds}

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python genetic_algorithm.py \
    -data_folder $data_folder \
    -save_dir $output_root \
    -idx_start $idx_start \
    -population_size $pop \
    -cross_rate $cross \
    -mutate_rate $mutate \
    -rounds $rounds \
    -num_instances $num \
    -max_n $max_n \
    -min_n $min_n \
    -min_c $min_c \
    -max_c $max_c \
    -n_graphs 5000 \
    -er_p $p \
    -base_model_dump $base_model_dump \
    $@
