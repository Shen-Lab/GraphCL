#!/bin/bash

min_n=90
max_n=100
p=0.02
output_root=../../dropbox/data/components

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

for t_c in 1 2 3 4 5; do

n_comp=$t_c

python gen_er_components.py \
    -save_dir $output_root \
    -max_n $max_n \
    -min_n $min_n \
    -num_graph 5000 \
    -p $p \
    -n_comp $n_comp

done
