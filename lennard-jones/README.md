# Learning to grow: Lennard-jones potential

The objective is to maximize Q6 bond-order parameter\
Fixed values of epsilon and sigma parameters\
Varied values of temperature and pressure\
Solution is an array of real values that represent a neural network

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
python3.8 scripts/run_l2g.py -gpus <number of gpus>
conda deactivate
```

## Reference

S. Whitelam, I. Tamblyn. "Learning to grow: control of materials self-assembly using evolutionary reinforcement learning". Phys. Rev. E, 2020. DOI: 10.1103/PhysRevE.101.052604 
