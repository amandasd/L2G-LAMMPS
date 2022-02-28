#!/usr/bin/env python

from numpy.random import randint

import numpy as np
import os,sys

#################################################################################
# LAMMPS parameters
#################################################################################

nve_steps = 1000000
npt_steps = 50000
n_steps = 10 #number total of steps = n_steps * npt_steps + (nve_steps + npt_steps)
dump_freq = 10000

#################################################################################
# end of parameters
#################################################################################

#################################################################################
# LAMMPS functions
#################################################################################

# run LAMMPS for all candidates in the population
def run_lammps(temp, press, state, gen, n_pop, n_gpus):
    # template to generate the LAMMPS input files for each candidate in the population
    filein = "input/He-in.restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for each candidate in the population
    for p in range(n_pop):

        newdata = filedata.replace("variable                T equal 375","variable                T equal {}".format(temp[p]))
        newdata = newdata.replace("variable                P equal 150000","variable                P equal {}".format(press[p]))
        newdata = newdata.replace("variable                npt_steps equal 10000","variable                npt_steps equal {}".format(npt_steps))
        newdata = newdata.replace("dump            1 all custom 10000 He.dump id type x y z c_CNA","dump            1 all custom {} output/He-{}.dump id type x y z c_CNA".format(dump_freq,p))
        newdata = newdata.replace("restart         ${npt_steps} He.restart","restart         {} output/He-{}.restart".format(npt_steps,p))

        if state > 0:
            newdata = newdata.replace("read_restart    input/He.restart.1000000","read_restart    output/He-"+str(p)+".restart."+str(state))

        fileout = "input/in."+str(p)
        f = open(fileout,'w')
        f.write(newdata)
        f.close()

    # run LAMMPS for each candidate in the population
    os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(n_gpus)+' 0'+' '+str(gen))


# run LAMMPS for the best candidate in the population
def best_lammps(temp, press, state, gen):
    # template to generate the LAMMPS input files for each candidate in the population
    filein = "input/He-in.restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for the best candidate in the population
    newdata = filedata.replace("variable                T equal 375","variable                T equal {}".format(temp[0]))
    newdata = newdata.replace("variable                P equal 150000","variable                P equal {}".format(press[0]))
    newdata = newdata.replace("variable                npt_steps equal 10000","variable                npt_steps equal {}".format(npt_steps))
    newdata = newdata.replace("dump            1 all custom 10000 He.dump id type x y z c_CNA","dump            1 all custom {} output/He-best.dump id type x y z c_CNA".format(dump_freq))
    newdata = newdata.replace("restart         ${npt_steps} He.restart","restart         {} output/He-best.restart".format(npt_steps))

    if state > 0:
        newdata = newdata.replace("read_restart    input/He.restart.1000000","read_restart    output/He-best.restart."+str(state))

    fileout = "input/in.best"
    f = open(fileout,'w')
    f.write(newdata)
    f.close()

    # run LAMMPS for each candidate in the population
    os.system('./scripts/run_lammps.sh 1 1 1 -1')
    

def get_scores(gen, n, n_iter):
    scores = []
    for p in range(n):
        if gen <= n_iter:
            filein = "output/out-"+str(gen)+"."+str(p)
        else:
            filein = "output/scores-best.txt"
        f = open(filein,'r')
        lines = f.readlines()
        f.close()
        matches = [line for line in lines if "Loop time" in line]
        index = lines.index(matches[0])
        scores.append(float(' '.join(lines[index-1].split()).split(' ')[7]))
    return scores


def delete_output_files(gen, n, n_iter):
    ini_state = npt_steps+nve_steps
    end_state = n_steps*npt_steps+npt_steps+nve_steps+nve_steps
    # delete restart files
    for i in np.arange(ini_state,end_state,1000):
        if gen <= n_iter:
            os.system('rm -f output/He-*.restart.'+str(i))
        else:
            os.system('rm -f output/He-best.restart.'+str(i))
    # delete dump and out files
    if gen <= n_iter:
        os.system('rm -f output/He-*.dump')
        #os.system('rm -f output/out.*')

#################################################################################
# end of functions
#################################################################################
