#!/bin/bash

min_n=40
max_n=50
p=0.05
dropbox=../../dropbox/
data_folder=$dropbox/data/components
min_c=1
max_c=3
max_lv=4

save_fold=nodes-${min_n}-${max_n}-p-${p}-c-${min_c}-${max_c}-lv-${max_lv}
output_root=$HOME/scratch/results/graph_classification/components/$save_fold
saved_model=$output_root/epoch-best

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

python er_components.py \
    -data_folder $data_folder \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -max_lv $max_lv \
    -min_c $min_c \
    -max_c $max_c \
    -saved_model $saved_model \
    -n_graphs 5000 \
    -er_p $p \
    $@
