#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH --nodes 1
#SBATCH -G 8
#SBATCH -n 40
#SBATCH -c 2
#SBATCH --switches=1
#SBATCH --exclusive     
#SBATCH -J learningToGrow
#SBATCH --output=l2g.out

ROOT_PATH="/global/homes/a/asdufek/projects/digital-twin/learningTogrow/lennard-jones"

module purge 
module load cgpu
module load cmake 
module load PrgEnv-llvm/12.0.0-git_20210117

module load python

export OMP_NUM_THREADS=1

date
python3.8 $ROOT_PATH/scripts/run_l2g.py -gpus 8
date
