#!/bin/bash -ex
# 0 - 24

i1=$2
i2=$(( $i1 + 4 ))
for i in $(seq $i1 $i2)
do
    ./run.sh $1 $i $3
done

