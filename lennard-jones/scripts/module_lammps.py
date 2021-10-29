#!/usr/bin/env python

from numpy.random import randint
import random

import numpy as np
import os,sys

#################################################################################
# LAMMPS parameters
#################################################################################

bounds = [[0.4, 2],  [0.5, 1]] # define range for temperature and pressure: t_min, t_max, p_min, p_max
nve_steps = 1000
npt_steps = 5000
n_steps = 3 #number total of steps = n_steps * npt_steps + (nve_steps + npt_steps)

#################################################################################
# end of parameters
#################################################################################

#################################################################################
# LAMMPS functions
#################################################################################

# run LAMMPS for all candidates in the population
def run_lammps(temp, press, state, gen, n_pop, n_gpus):
    # template to generate the LAMMPS input files for each candidate in the population
    if state == 0:
        filein = "input/in.original"
    else:
        filein = "input/in.restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for each candidate in the population
    for p in range(n_pop):
        # velocity all create <temp> <seed> dist <gaussian> ... 0 < seed <= 8 digits 
        newdata = filedata.replace("velocity all create 1.0 87287 dist gaussian","velocity all create 1.0 "+str(randint(0, 99999999))+" dist gaussian")
        newdata = newdata.replace("output.xyz","output/data-"+str(p)+".xyz")
        newdata = newdata.replace("orderparameter_timeave.txt","output/scores-"+str(p)+".txt")
        newdata = newdata.replace("restart 6000 output/lj.restart","restart "+str(nve_steps+npt_steps)+" output/restart-"+str(p))
        newdata = newdata.replace("fix 2 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 2 all npt temp "+str(temp[p])+" "+str(temp[p])+" "+str(temp[p])+" iso "+str(press[p])+" "+str(press[p])+" "+str(press[p]))
        newdata = newdata.replace("# run steps in the NVE ensemble\nrun 1000", "# run steps in the NVE ensemble\nrun "+str(nve_steps))
        newdata = newdata.replace("# run more steps in the NPT ensemble\nrun 5000", "# run more steps in the NPT ensemble\nrun "+str(npt_steps))

        if state > 0:
            newdata = newdata.replace("read_restart output/lj.restart.6000","read_restart output/restart-"+str(p)+"."+str(state))
            newdata = newdata.replace("fix 1 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 1 all npt temp "+str(temp[p])+" "+str(temp[p])+" "+str(temp[p])+" iso "+str(press[p])+" "+str(press[p])+" "+str(press[p]))
            newdata = newdata.replace("restart 1000 output/lj.restart","restart 1000 output/restart-"+str(p))
    
        fileout = "input/in."+str(p)
        f = open(fileout,'w')
        f.write(newdata)
        f.close()

    # run LAMMPS for each candidate in the population
    os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(n_gpus)+' 0')


# run LAMMPS for the best candidate in the population
def best_lammps(temp, press, state, gen):
    # template to generate the LAMMPS input files for each candidate in the population
    if state == 0:
        filein = "input/in.original"
    else:
        filein = "input/in.restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for the best candidate in the population
    # velocity all create <temp> <seed> dist <gaussian> ... 0 < seed <= 8 digits 
    newdata = filedata.replace("velocity all create 1.0 87287 dist gaussian","velocity all create 1.0 "+str(randint(0, 99999999))+" dist gaussian")
    newdata = newdata.replace("output.xyz","output/data-best.xyz")
    newdata = newdata.replace("orderparameter_timeave.txt","output/scores-best.txt")
    newdata = newdata.replace("restart 6000 output/lj.restart","restart "+str(nve_steps+npt_steps)+" output/restart-best")
    newdata = newdata.replace("fix 2 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 2 all npt temp "+str(temp[0])+" "+str(temp[0])+" "+str(temp[0])+" iso "+str(press[0])+" "+str(press[0])+" "+str(press[0]))
    newdata = newdata.replace("# run steps in the NVE ensemble\nrun 1000", "# run steps in the NVE ensemble\nrun "+str(nve_steps))
    newdata = newdata.replace("# run more steps in the NPT ensemble\nrun 5000", "# run more steps in the NPT ensemble\nrun "+str(npt_steps))

    if state > 0:
        newdata = newdata.replace("read_restart output/lj.restart.6000","read_restart output/restart-best."+str(state))
        newdata = newdata.replace("fix 1 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 1 all npt temp "+str(temp[0])+" "+str(temp[0])+" "+str(temp[0])+" iso "+str(press[0])+" "+str(press[0])+" "+str(press[0]))
        newdata = newdata.replace("restart 1000 output/lj.restart","restart 1000 output/restart-best")
    
    fileout = "input/in.best"
    f = open(fileout,'w')
    f.write(newdata)
    f.close()

    # run LAMMPS for each candidate in the population
    os.system('./scripts/run_lammps.sh 1 1 1')
    

def get_scores(gen, n, n_iter):
    scores = []
    for p in range(n):
        if gen <= n_iter:
            filein = "output/scores-"+str(p)+".txt"
        else:
            filein = "output/scores-best.txt"
        f = open(filein,'r')
        lines = f.readlines()
        f.close()
        scores.append(float(lines[2].split(' ')[1]))
    return scores


# initialize temperature and pressure values: 0 (random), 1 (fixed values), 2 (mutated from a given value)
def initialize_T_P(n, opt, vtemp=None, vpress=None):
    if opt == 0:
        temp  = np.random.uniform(bounds[0][0], bounds[0][1], n)
        press = np.random.uniform(bounds[1][0], bounds[1][1], n)
    elif opt == 1:
        temp  = np.full(n, vtemp, dtype=float)
        press = np.full(n, vpress, dtype=float)
    elif opt == 2:
        for p in range(n):
            temp[p] = vtemp + random.gauss(0,0.01)
            if temp[p] > bounds[0][1]:
                temp[p] = bounds[0][1]
            if temp[p] < bounds[0][0]:
                temp[p] = bounds[0][0]
            press[p] = vpress + random.gauss(0,0.01)
            if press[p] > bounds[1][1]:
                press[p] = bounds[1][1]
            if press[p] < bounds[1][0]:
                press[p] = bounds[1][0]
    else:
        print("Valid options: 0 (random), 1 (mutated from a given value), 2 (fixed values)")

    return temp, press


def delete_output_files(state, gen, n, n_iter):
    for p in range(n):
        # delete restart files
        ini_state = npt_steps+nve_steps
        end_state = n_steps*npt_steps+npt_steps+nve_steps
        if state > 0 and state > (nve_steps+npt_steps):
            for i in np.arange(ini_state,end_state+nve_steps,1000):
                if gen <= n_iter:
                    os.system('rm output/restart-'+str(p)+"."+str(i))
                else:
                    os.system('rm output/restart-best.'+str(i))
        elif state > 0:
            if gen <= n_iter:
                os.system('rm output/restart-'+str(p)+"."+str(state))
            else:
                os.system('rm output/restart-best.'+str(state))
        # delete xyz and out files
        os.system('rm output/data-'+str(p)+'.xyz')
        os.system('rm output/out.'+str(p))

#################################################################################
# end of functions
#################################################################################
