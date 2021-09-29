#!/bin/bash


#------------old cleanup-----------------


rm slurm_lammps_parallel flag


#-----------get the input---------------

npop=$1
opt=$2


#-----Create a submission script based on the opt values-----


#source ~/anaconda3/bin/activate
source /global/cfs/cdirs/m1917/blast_ff/activate_py

python JobCreator.py $npop $opt

sbatch slurm_lammps_parallel 

#------wait till the job is finished-------


while true
do
        sleep 2
        if [[ -f flag ]] ;
        then
		break
	fi
done







