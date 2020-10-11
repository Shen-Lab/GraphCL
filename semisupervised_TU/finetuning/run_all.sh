#!/bin/bash -ex
# 0 - 24

i1=$2
i2=$(( $i1 + 4 ))
for i in $(seq $i1 $i2)
do
  ./run.sh $1 $i 0 $3 $4
  ./run.sh $1 $i 1 $3 $4
  ./run.sh $1 $i 2 $3 $4
  ./run.sh $1 $i 3 $3 $4
  ./run.sh $1 $i 4 $3 $4
done
