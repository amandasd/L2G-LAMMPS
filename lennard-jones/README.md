# Learning to Grow for LAMMPS

Solution is an array of real values that represents a neural network
Changes must be made to the module\_lammps.py file according to your LAMMPS simulation

## Run on Cori GPU

Define the `LAMMPS_DIR` environment variable in `scripts/submit.sh` according to the directory where your `lmp` executable is located.
```
sbatch ./scripts/submit.sh
```

or

```
module purge 
module load cgpu 
module load cmake 
module load PrgEnv-llvm/12.0.0-git_20210117
module load python/3.8-anaconda-2020.11 

export LAMMPS_DIR=<lmp executable directory>
export OMP_NUM_THREADS=1

salloc -C gpu -N <number of nodes> -G <number of gpus> -t <time> -A <account> --exclusive -q special

source activate myenv-3.8
python3.8 scripts/run_l2g.py -help
python3.8 scripts/run_l2g.py -gpus <number of gpus> -gen <number of generations> -pop <population size> -mr <mutation rate> -ms <mutation sigma> -ts <tournament size> -best <number of retained solutions> -elitism -hid <number of hidden nodes> -restart -tmin <minimum temperature> -tmax <maximum temperature> -pmin <minimum pressure> -pmax <maximum pressure> -opt <option to initialize temperature and pressure> -vtemp <initial temperature> -vpress <initial pressure> -tf <temperature factor> -pf <pressure factor>
conda deactivate
```

## Reference

#### Paper

S. Whitelam, I. Tamblyn. "Learning to grow: control of materials self-assembly using evolutionary reinforcement learning". Phys. Rev. E, 2020. DOI: 10.1103/PhysRevE.101.052604 

#### Site

https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
