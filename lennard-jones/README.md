# Learning to grow: Lennard-jones potential

Objective: maximize Q6 bond-order parameter
Fixed values of epsilon and sigma 
Varied values of temperature and pressure
Solution is an array of real values that represent a neural network

## Environment variable

Define the `LAMMPS_DIR` environment variable in `scripts/submit.sh` according to the directory where your `lmp` executable is located.

## Run on Cori GPU

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

salloc -C gpu -N 1 -G 8 -t 20 -A m1759 --exclusive -q special

source activate myenv-3.8
python3.8 scripts/run_l2g.py -gpus <number of gpus>
python3.8 scripts/run_l2g.py -gpus 8
conda deactivate
```


