#!/usr/bin/env python

from numpy.random import randint
from numpy.random import rand
import random

from datetime import datetime 

import numpy as np
import os,sys
import argparse

#################################################################################
# Learning To Grow: Lennard-Jones potential
# Objective: maximize Q6 bond-order parameter
# Fixed values of epsilon and sigma 
# Varied values of T and P
# Solution is an array of real values that represent a neural network
#################################################################################
# usage: run_l2g.py [-h] [-gpus NUMBER_OF_GPUS]
#
# optional arguments:
#  -h, --help                                  show this help message and exit
#  -gpus, --number-of-gpus NUMBER_OF_GPUS 
#                                              number of gpus [default=1]
# Example:
#    python run_l2g.py -gpus 8
#################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-gpus", "--number-of-gpus", type=int, default=1, help="number of nodes [default=1]")
parser.add_argument("-measure", "--measure-of-assembly", type=int, default=1, help="measure of assembly [default=1]: 1 (Q6 bond-order parameter) or 2 (number of contacts per particle)")
args = parser.parse_args()

#################################################################################
# parameters
#################################################################################

# define Lennard-Jones parameters
bounds = [[0.4, 2],  [0.5, 1]] # define range for temperature and pressure: t_min, t_max, p_min, p_max
lj_epsilon = 4.77
lj_sigma = 1.0

# define LAMMPS parameters
nve_steps = 1000
npt_steps = 5000
n_steps = 3 #number total of steps = n_steps * npt_steps + (nve_steps + npt_steps)

# define evolutionary algorithm parameters
n_iter = 3 # define the total iterations
n_pop = 16 # define the population size
r_mut = 0.9 # mutation rate
n_best = max([4, int(np.ceil(n_pop * 0.1))]) # best solutions
elitism = True

# define neural network parameters 
input_layer  = 1 
hidden_layer = 10 
output_layer = 2

#################################################################################
# end of parameters
#################################################################################

#################################################################################
# functions
#################################################################################

# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
    	    selection_ix = ix
    return pop[selection_ix]

# mutation operator
def mutation(ind, mu, sigma, r_mut):
    for i in range(len(ind)):
        # check for a mutation
        if rand() < r_mut:
            ind[i] += random.gauss(mu,sigma)
        else:
            ind[i] = random.gauss(0,1)
#TODO: do I need to check bounds?

# run LAMMPS for all candidates in the population
def run_lammps(temp, press, state, gen):
    # template to generate the LAMMPS input files for each candidate in the population
    if state == 0:
        filein = "input/in.lennard-jones-original"
    else:
        filein = "input/in.lennard-jones-original-restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for each candidate in the population
    for p in range(n_pop):
        # velocity all create <temp> <seed> dist <gaussian> ... 0 < seed <= 8 digits 
        newdata = filedata.replace("velocity all create 1.0 87287 dist gaussian","velocity all create 1.0 "+str(randint(0, 99999999))+" dist gaussian")
        newdata = newdata.replace("output.xyz","output/data.lennard-jones-"+str(p)+".xyz")
        newdata = newdata.replace("orderparameter_timeave.txt","output/scores-"+str(p)+".txt")
        newdata = newdata.replace("restart 6000 output/lj.restart","restart "+str(nve_steps+npt_steps)+" output/lj.restart-"+str(p))
        newdata = newdata.replace("fix 2 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 2 all npt temp "+str(temp[p])+" "+str(temp[p])+" "+str(temp[p])+" iso "+str(press[p])+" "+str(press[p])+" "+str(press[p]))
        newdata = newdata.replace("# run steps in the NVE ensemble\nrun 1000", "# run steps in the NVE ensemble\nrun "+str(nve_steps))
        newdata = newdata.replace("# run more steps in the NPT ensemble\nrun 5000", "# run more steps in the NPT ensemble\nrun "+str(npt_steps))

        if state > 0:
            newdata = newdata.replace("read_restart output/lj.restart.6000","read_restart output/lj.restart-"+str(p)+"."+str(state))
            newdata = newdata.replace("fix 1 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 1 all npt temp "+str(temp[p])+" "+str(temp[p])+" "+str(temp[p])+" iso "+str(press[p])+" "+str(press[p])+" "+str(press[p]))
            newdata = newdata.replace("restart 1000 output/lj.restart","restart 1000 output/lj.restart-"+str(p))
    
        fileout = "input/in.lennard-jones-"+str(p)
        f = open(fileout,'w')
        f.write(newdata)
        f.close()

    # run LAMMPS for each candidate in the population
    #print("Start running LAMMPS["+str(gen)+"]")
    os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(args.number_of_gpus)+' 0')
    #print("End running LAMMPS["+str(gen)+"]")
    #print()
    
    for p in range(n_pop):
        os.system('rm output/data.lennard-jones-'+str(p)+'.xyz')
        os.system('rm output/out.lennard-jones-'+str(p))
        if state > 0 and state > (nve_steps+npt_steps):
            for i in np.arange(state-(npt_steps-nve_steps),state+nve_steps,1000):
                os.system('rm output/lj.restart-'+str(p)+"."+str(i))
        elif state > 0:
            os.system('rm output/lj.restart-'+str(p)+"."+str(state))

# run LAMMPS for the best candidate in the population
def best_lammps(temp, press, state, gen):
    # template to generate the LAMMPS input files for each candidate in the population
    if state == 0:
        filein = "input/in.lennard-jones-original"
    else:
        filein = "input/in.lennard-jones-original-restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for the best candidate in the population
    # velocity all create <temp> <seed> dist <gaussian> ... 0 < seed <= 8 digits 
    newdata = filedata.replace("velocity all create 1.0 87287 dist gaussian","velocity all create 1.0 "+str(randint(0, 99999999))+" dist gaussian")
    newdata = newdata.replace("output.xyz","output/data.lennard-jones-best.xyz")
    newdata = newdata.replace("orderparameter_timeave.txt","output/scores-best.txt")
    newdata = newdata.replace("restart 6000 output/lj.restart","restart "+str(nve_steps+npt_steps)+" output/lj.restart-best")
    newdata = newdata.replace("fix 2 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 2 all npt temp "+str(temp[0])+" "+str(temp[0])+" "+str(temp[0])+" iso "+str(press[0])+" "+str(press[0])+" "+str(press[0]))
    newdata = newdata.replace("# run steps in the NVE ensemble\nrun 1000", "# run steps in the NVE ensemble\nrun "+str(nve_steps))
    newdata = newdata.replace("# run more steps in the NPT ensemble\nrun 5000", "# run more steps in the NPT ensemble\nrun "+str(npt_steps))

    if state > 0:
        newdata = newdata.replace("read_restart output/lj.restart.6000","read_restart output/lj.restart-best."+str(state))
        newdata = newdata.replace("fix 1 all npt temp 1.0 1.0 1.0 iso 1.0 1.0 1.0","fix 1 all npt temp "+str(temp[0])+" "+str(temp[0])+" "+str(temp[0])+" iso "+str(press[0])+" "+str(press[0])+" "+str(press[0]))
        newdata = newdata.replace("restart 1000 output/lj.restart","restart 1000 output/lj.restart-best")
    
    fileout = "input/in.lennard-jones-best"
    f = open(fileout,'w')
    f.write(newdata)
    f.close()

    # run LAMMPS for each candidate in the population
    #print("Start running LAMMPS["+str(gen)+"]")
    os.system('./scripts/run_lammps.sh 1 1 1')
    #print("End running LAMMPS["+str(gen)+"]")
    #print()
    
    if state > 0 and state > (nve_steps+npt_steps):
        for i in np.arange(state-(npt_steps-nve_steps),state+nve_steps,1000):
            os.system('rm output/lj.restart-best.'+str(i))
    elif state > 0:
        os.system('rm output/lj.restart-best.'+str(state))

# run neural networks
def run_networks(pop, temp, press, node_input, n):

    for p in range(n):

        # get weights and bias
        bias_hidden = pop[p][:hidden_layer]
        weight_ih = pop[p][hidden_layer:hidden_layer+input_layer*hidden_layer]
        weight_ih = np.reshape(weight_ih,(input_layer,hidden_layer))
        weight_ho = pop[p][hidden_layer+input_layer*hidden_layer:hidden_layer+input_layer*hidden_layer+output_layer*hidden_layer]
        weight_ho = np.reshape(weight_ho,(hidden_layer,output_layer))
 
        # calculate Sj 
        node_hidden = np.zeros(shape=(hidden_layer), dtype=float)
        for j in range(hidden_layer):
            node_hidden[j] = np.sum(np.dot(node_input,weight_ih[:,j]))
        node_hidden = np.tanh(np.add(node_hidden,bias_hidden))
        
        # calculate Sk 
        node_output = np.zeros(shape=(output_layer), dtype=float)
        for k in range(output_layer):
            node_output[k] = np.sum(np.dot(node_hidden,weight_ho[:,k]))/(1.*hidden_layer)

        temp[p] += node_output[0]
        if temp[p] > bounds[0][1]:
            temp[p] = bounds[0][1]
        if temp[p] < bounds[0][0]:
            temp[p] = bounds[0][0]
        
        press[p] += node_output[1]
        if press[p] > bounds[1][1]:
            press[p] = bounds[1][1]
        if press[p] < bounds[1][0]:
            press[p] = bounds[1][0]

    return temp, press

# evaluate all candidates in the population: run neural networks and LAMMPS
def evaluate(pop, gen, n):

    #TODO: for a new generation, are temp/press random values? Or are they the last values used in the simulation? Or are they fixed values?
    if gen < 0:
        temp  = np.random.uniform(bounds[0][0], bounds[0][1], n)
        press = np.random.uniform(bounds[1][0], bounds[1][1], n)
    else:
        for p in range(n):
            temp[p] += random.gauss(0,0.01)
            if temp[p] > bounds[0][1]:
                temp[p] = bounds[0][1]
            if temp[p] < bounds[0][0]:
                temp[p] = bounds[0][0]
            press[p] += random.gauss(0,0.01)
            if press[p] > bounds[1][1]:
                press[p] = bounds[1][1]
            if press[p] < bounds[1][0]:
                press[p] = bounds[1][0]

    # run LAMMPS
    if gen < n_iter:
        run_lammps(temp, press, 0, gen) 
    else:
        best_lammps(temp, press, 0, n_iter)

    for s in range(n_steps):
        node_input = s * 1./n_steps 
        # run neural networks
        temp, press = run_networks(pop, temp, press, node_input, n)
        state = s*npt_steps+npt_steps+nve_steps
        # run LAMMPS
        if gen < n_iter:
            run_lammps(temp, press, state, gen) 
        else:
            best_lammps(temp, press, state, n_iter)

    for p in range(n):
        state = n_steps*npt_steps+npt_steps+nve_steps
        if state > 0 and state > (nve_steps+npt_steps):
            for i in np.arange(state-(npt_steps-nve_steps),state+nve_steps,1000):
                if gen < n_iter:
                    os.system('rm output/lj.restart-'+str(p)+"."+str(i))
                else:
                    os.system('rm output/lj.restart-best.'+str(i))
        elif state > 0:
            if gen < n_iter:
                os.system('rm output/lj.restart-'+str(p)+"."+str(state))
            else:
                os.system('rm output/lj.restart-best.'+str(state))

    scores = []
    for p in range(n):
        if gen < n_iter:
            filein = "output/scores-"+str(p)+".txt"
        else:
            filein = "output/scores-best.txt"
        f = open(filein,'r')
        lines = f.readlines()
        f.close()
        if args.measure_of_assembly == 1: #Q6 bond-order parameter
            scores.append(float(lines[2].split(' ')[1])) 
        elif args.measure_of_assembly == 2: #number of contacts per particle
            scores.append(float(lines[2].split(' ')[2])) 
        else:
            print("Valid measure of assembly: 1 (Q6 bond-order parameter) or 2 (number of contacts per particle)")
    return scores

#################################################################################
# end of functions
#################################################################################

#################################################################################
# learning to grow: main code
#################################################################################

print()
print("population size: "+str(n_pop))
print("number of retained solutions: "+str(n_best)) 
print("number of iterations: "+str(n_iter))
print()

random.seed(datetime.now())

# generate a random initial population: weights and bias of neural networks
pop = [[random.gauss(0,1) for _ in range(hidden_layer+input_layer*hidden_layer+output_layer*hidden_layer)] for _ in range(n_pop)]

# evaluate all candidates in the population: run neural networks and LAMMPS
scores = evaluate(pop, -1, n_pop)

# select the best candidate 
idx = scores.index(max(scores))
best, best_eval = pop[idx], scores[idx]
print(">-1, new best = %f" % (best_eval))
print()

for gen in range(n_iter): # maximum number of iterations

    # rank the scores 
    indices = [scores.index(x) for x in sorted(scores, reverse=True)]

    # select parents from the current population
    # n_best candidates are selected to generate new candidates
    selected = [selection(np.take(pop,indices,0)[:n_best], np.take(scores,indices,0)[:n_best]) for _ in range(n_pop)]
    
    # create the next generation
    children = list()
    for i in range(0, n_pop):
        #copy the best candidate to next generation without mutation
        if elitism:
            children.append(pop[idx])
            elitism = False
            continue
        c = selected[i]
        # mutation: change weights and bias of neural networks
        mutation(c, 0, 0.01, r_mut)
        # store for next generation
        children.append(c)
    # replace population
    pop = children

    # evaluate all candidates in the population: run neural networks and LAMMPS
    scores = evaluate(pop, gen, n_pop)

    # select the best candidate
    idx = scores.index(max(scores))
    if scores[idx] > best_eval:
        best, best_eval = pop[idx], scores[idx]
        print(">%d, new best = %f" % (gen, best_eval))
        print()

# evaluate the best candidate
scores = evaluate([pop[idx]], n_iter, 1)
print("best = %f" % (scores[0]))

#################################################################################
# end of main code
#################################################################################
