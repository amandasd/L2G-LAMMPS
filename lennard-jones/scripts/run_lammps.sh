#!/bin/bash

ROOT_PATH="/global/homes/a/asdufek/projects/digital-twin/learningTogrow/lennard-jones"

npop=$1
ngpus=$2
opt=$3

#gpu
p=0
while [ $p -lt $npop ]; do
   for g in $(seq 0 $(( $ngpus - 1))); do
      if [ $p -lt $npop ]; then
	 #echo "Running pop "$p "gpu "$g
	 if [ $opt -eq 1 ]; then
            ./lmp -pk gpu $ngpus gpuID $g -sf gpu -in $ROOT_PATH/input/in.lennard-jones-best > $ROOT_PATH/output/out.lennard-jones-best &
         elif [ $opt -eq 0 ]; then
            ./lmp -pk gpu $ngpus gpuID $g -sf gpu -in $ROOT_PATH/input/in.lennard-jones-$p > $ROOT_PATH/output/out.lennard-jones-$p &
         fi
      fi
      p=$(( $p + 1 ))
   done
   #echo "waiting..."
   wait
done
