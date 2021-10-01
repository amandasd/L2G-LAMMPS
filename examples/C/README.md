This is the LearningToGrow module that attempts to learn a protocol to synthesize particular phase of Carbon. To run the code follow these steps:
1) Run pre_run.sh to clean up the directory using
./pre_run.sh
2) Set desired variables in the run_l2g.py. These include number of GA generations, population size, temp and pressure bound, etc. Then run using
python ./pre_run.sh

Important notes:
Necessary modifications maybe be necessary for the following:
1) conda environment that has fingerprinting modules (SOAP) is need (see line 20 in run_lammps.sh)
2) Cori Account info will have to be modified (see line 27 in JobCreator.py)