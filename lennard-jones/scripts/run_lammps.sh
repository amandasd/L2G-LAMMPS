#!/bin/bash

HOME="$(pwd)"
OUTPUT_DIR="$HOME/output"

mkdir -p "$OUTPUT_DIR"

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
            $LAMMPS_DIR/lmp -pk gpu $ngpus gpuID $g -sf gpu -in $HOME/input/in.best > $OUTPUT_DIR/out.best &
         elif [ $opt -eq 0 ]; then
            $LAMMPS_DIR/lmp -pk gpu $ngpus gpuID $g -sf gpu -in $HOME/input/in.$p > $OUTPUT_DIR/out.$p &
         fi
      fi
      p=$(( $p + 1 ))
   done
   #echo "waiting..."
   wait
done
