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
#SBATCH -J L2G_MC
#SBATCH --output=l2g_mc.out

HOME="$(pwd)"
PYTHONDIR="/global/homes/a/asdufek/.conda/envs/polymorph/bin"

module purge
module load cgpu
module load cmake
module load PrgEnv-llvm/12.0.0-git_20210117
module load python/3.9-anaconda-2021.11

export LAMMPS_DIR=$HOME
export OMP_NUM_THREADS=1

date
#T and P -> high score
#$PYTHONDIR/python3.9 $HOME/scripts/run_l2g.py -s mc -gpus 8 -sim 16                    -gen 32 -tmin 200 -tmax 400 -pmin 100000 -pmax 400000 -opt 1 -vtemp 200 -vpress 355467 -tf 100 -pf 150000 -ms 0.3 -hid 1000 -input 3 -odir '/global/cscratch1/sd/asdufek/project/digital-twin/learningTogrow/output/mc-bcc-high-i3'
$PYTHONDIR/python3.9 $HOME/scripts/run_l2g.py -s ga -gpus 8 -pop 16 -popf 3 -best 8 -e -gen 2 -tmin 200 -tmax 400 -pmin 100000 -pmax 400000 -opt 1 -vtemp 200 -vpress 355467 -tf 100 -pf 150000 -ms 0.3 -hid 1000 -input 3
#T and P -> low score
#$PYTHONDIR/python3.9 $HOME/scripts/run_l2g.py -s mc -gpus 8 -sim 16                    -gen 40 -tmin 200 -tmax 400 -pmin 100000 -pmax 400000 -opt 1 -vtemp 385 -vpress 118319 -tf 100 -pf 150000 -ms 0.3 -hid 1000 -input 3 -odir '/global/cscratch1/sd/asdufek/project/digital-twin/learningTogrow/output/mc-bcc-low-i3'
#$PYTHONDIR/python3.9 $HOME/scripts/run_l2g.py -s ga -gpus 8 -pop 16 -popf 3 -best 8 -e -gen 32 -tmin 200 -tmax 400 -pmin 100000 -pmax 400000 -opt 1 -vtemp 385 -vpress 118319 -tf 100 -pf 150000 -ms 0.3 -hid 1000 -input 3
date
