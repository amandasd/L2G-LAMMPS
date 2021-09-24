#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
from glob import glob
import sys

npop = int(sys.argv[1])
opt = int(sys.argv[2])

#---------the filename is slurm_lammps_parallel -------------


# In[10]:


#---- job script generator----------

def job_script(outfile,npop,opt):
    
    
    with open(outfile,"a") as outfile:
        outfile.write("#!/bin/bash\n")
        outfile.write("#SBATCH -A m3793\n")
        outfile.write("#SBATCH -N {}\n".format(npop))
        outfile.write("#SBATCH -C knl\n")
        outfile.write("#SBATCH -q debug\n")
        outfile.write("#SBATCH -J l2g_lammps\n")
        outfile.write("#SBATCH -t 00:05:00\n")
        
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
                outfile.write("srun -N 1 -n 64 -c 1 --cpu-bind=cores lmp_cori  -in $HOME/input/in.lennard-jones-best> $OUTPUT_DIR/out.lennard-jones-best &\n")
                outfile.write("\n")
            else: 
                outfile.write("srun -N 1 -n 64 -c 1 --cpu-bind=cores lmp_cori  -in $HOME/input/in.lennard-jones-{}> $OUTPUT_DIR/out.lennard-jones-best-{} &\n".format(i,i))
                outfile.write("\n")
                
        outfile.write("wait\n")
        
        outfile.write("touch flag")
        
        


# In[11]:


job_script("slurm_lammps_parallel",npop,opt)


# In[ ]:





# In[ ]:




