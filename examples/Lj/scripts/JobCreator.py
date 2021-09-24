#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
from glob import glob
import sys

npop = sys.argv[1]
opt = sys.argv[2]

#---------the filename is slurm_lammps_parallel -------------


# In[10]:


#---- job script generator----------

def job_script(outfile,npop,opt):
    
    
    with open(outfile,"a") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("#SBATCH -A m3793\n")
        outfile.write("#SBATCH -N {}\n".format(npop))
        outfile.write("#SBATCH -C debug\n")
        outfile.write("#SBATCH -q debug\n")
        outfile.write("#SBATCH -J l2g_lammps\n")
        outfile.write("#SBATCH -t 00:10:00\n")
        
        outfile.write("\n\n")
        
        outfile.write("HOME=\"$(pwd)\"\n")
        outfile.write("OUTPUT_DIR=\"$HOME/output\"\n")
        outfile.write("mkdir -p \"$OUTPUT_DIR\"\n")
        outfile.write("\n\n")
        outfile.write("module load lammps\n")
        outfile.write("\n\n")
        

        outfile.write("export OMP_NUM_THREADS=1\n")
        outfile.write("export OMP_PLACES=threads\n")
        outfile.write("export OMP_PROC_BIND=spread\n")
        outfile.write("\n\n")
        
        
        for i in range(npop):
            if opt==1:
                outfile.write("srun -N 1 -n 64 -c 1 --cpu-bind=cores lmp_cori  -in $HOME/input/in.lennard-jones-best> $OUTPUT_DIR/input/in.lennard-jones-best &\n".format(os.getcwd()))
                outfile.write("\n")
            else: 
                outfile.write("srun -N 1 -n 64 -c 1 --cpu-bind=cores lmp_cori  -in $HOME/input/in.lennard-jones-best> $OUTPUT_DIR/input/input/in.lennard-jones-{}&\n".format(os.getcwd(),i))
                outfile.write("\n")
                
        outfile.write("wait\n")
        
        outfile.write("flag")
        
        


# In[11]:


job_script("slurm_lammps_parallel",npop,opt)


# In[ ]:





# In[ ]:




