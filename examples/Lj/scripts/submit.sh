#!/bin/bash
#SBATCH -A m3793
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -t 01:10:00
#SBATCH -J learningToGrow
#SBATCH --output=l2g.out

HOME="$(pwd)"

module purge 
module load cmake 
module load PrgEnv-llvm/12.0.0-git_20210117
module load python

export LAMMPS_DIR=$HOME
export OMP_NUM_THREADS=1

date
python $HOME/scripts/run_l2g.py
date
