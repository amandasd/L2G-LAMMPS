#!/bin/bash
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH --nodes 1
#SBATCH -G 8
#SBATCH -n 40
#SBATCH -c 2
#SBATCH --switches=1
#SBATCH --exclusive     
#SBATCH -J learningToGrow
#SBATCH --output=l2g.out

HOME="$(pwd)"

module purge 
module load cgpu
module load cmake 
module load PrgEnv-llvm/12.0.0-git_20210117
module load python/3.8-anaconda-2020.11

export LAMMPS_DIR=$HOME
export OMP_NUM_THREADS=1

date
python3.8 $HOME/scripts/run_l2g.py -gpus 8 -pop 16 -gen 8 -tmin 200 -tmax 400 -pmin 100000 -pmax 400000 -opt 0 -tf 10 -pf 15000 -ms 0.5 -best 8
#python3.8 $HOME/scripts/run_l2g.py -gpus 8 -pop 16 -gen 30 -tmin 200 -tmax 400 -pmin 100000 -pmax 400000 -opt 1 -vtemp 375 -vpress 150000 -tf 10 -pf 15000 -ms 0.5
date


