#!/usr/bin/env python

from numpy.random import randint
from numpy.random import rand
import random

from datetime import datetime 

import numpy as np
import os,sys
import argparse
from  getSoap import  Get_SOAP
from pymatgen.io.lammps.data import LammpsData
from pymatgen import Structure,Lattice

#################################################################################
# Learning To Grow: Lennard-Jones potential
# Objective: maximize Q6 bond-order parameter
# Fixed values of epsilon and sigma 
# Varied values of T and P
# Solution is an array of real values that represent a neural network

#################################################################################
# parameters
#################################################################################

# define LAMMPS parameters
bounds = [[1000, 2000],  [10000, 15000]] # define range for temperature and pressure: t_min, t_max, p_min, p_max
nve_steps = 1000
npt_steps = 5000
n_steps = 3 #number total of steps = n_steps * npt_steps + (nve_steps + npt_steps)

# define evolutionary algorithm parameters
n_iter = 3 # define the total iterations
n_pop = 16 # define the population size
r_mut = 0.9 # mutation rate
n_best = max([4, int(np.ceil(n_pop * 0.1))]) # best solutions

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

def SOAP(filename):
    struct = LammpsData.from_file(filename,atom_style="atomic").structure
    refStruct = LammpsData.from_file("in.Cubic",atom_style="atomic").structure
    soap = Get_SOAP(struct)
    ref_soap = Get_SOAP(refStruct)

    return np.linalg.norm(np.array(soap)-np.array(ref_soap))  #eucledian norm



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
        filein = "input/in.original"
    else:
        filein = "input/in.restart"
    f = open(filein,'r')
    filedata = f.read()
    f.close()

    # generate the LAMMPS input files for each candidate in the population
    for p in range(n_pop):
        # velocity all create <temp> <seed> dist <gaussian> ... 0 < seed <= 8 digits 
        newdata = filedata.replace("variable       P equal 1000.0","variable       P equal {}".format(press[p]))
        newdata = filedata.replace("variable       P equal 1000.0","variable       P equal {}".format(temp[p]))
        newdata = newdata.replace("restart       6000 output/data.restart","restart      "+str(nve_steps+npt_steps)+" output/data.restart-"+str(p))
        newdata = newdata.replace("dump          1 all custom 1000 dump.out id type x y z","dump          1 all custom 1000 dump-{}.out id type x y z".format(p)) 
        newdata = newdata.replace("write_data    out.data","write_data    out-{}.data".format(p))
    
        if state > 0:
            newdata = newdata.replace("read_restart   output/data.restart.6000","read_restart    output/lj.restart-"+str(p)+"."+str(state))
    
        fileout = "input/in."+str(p)
        f = open(fileout,'w')
        f.write(newdata)
        f.close()

    # run LAMMPS for each candidate in the population
    #print("Start running LAMMPS["+str(gen)+"]")
    os.system('bash ./run_lammps.sh '+str(n_pop)+' 0')
    #print("End running LAMMPS["+str(gen)+"]")
    #print()


    os.system('mv  out-* output')
    os.system('mv  dump-* output')
    
    for p in range(n_pop):

        if state > 0 and state > (nve_steps+npt_steps):
            for i in np.arange(state-(npt_steps-nve_steps),state+nve_steps,1000):
                os.system('rm output/data.restart-'+str(p)+"."+str(i))
        elif state > 0:
            os.system('rm output/data.restart-'+str(p)+"."+str(state))



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
    newdata = filedata.replace("variable       P equal 1000.0","variable       P equal {}".format(press[p]))
    newdata = filedata.replace("variable       P equal 1000.0","variable       P equal {}".format(temp[p]))
    newdata = newdata.replace("restart       6000 output/data.restart","restart      "+str(nve_steps+npt_steps)+" output/data.restart-best")
    newdata = newdata.replace("dump          1 all custom 1000 dump.out id type x y z","dump          1 all custom 1000 dump-{}.out id type x y z".format(p))
    newdata = newdata.replace("write_data    out.data","write_data    out-{}.data".format(p))
    
    
    if state > 0:
         newdata = newdata.replace("read_restart   output/data.restart.6000","read_restart    output/lj.restart-best."+str(state))
    fileout = "input/in.best"
    f = open(fileout,'w')
    f.write(newdata)
    f.close()

    # run LAMMPS for each candidate in the population
    #print("Start running LAMMPS["+str(gen)+"]")
    os.system('bash ./run_lammps.sh 1 1')
    #print("End running LAMMPS["+str(gen)+"]")
    #print()

    os.system('mv  out-* output')
    os.system('mv  dump-* output')
    
    if state > 0 and state > (nve_steps+npt_steps):
        for i in np.arange(state-(npt_steps-nve_steps),state+nve_steps,1000):
            os.system('rm output/data.restart-best.'+str(i))
    elif state > 0:
        os.system('rm output/data.restart-best.'+str(state))

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

# evaluate all candidates in the population

def evaluate(pop, gen, n):

    temp  = np.random.uniform(bounds[0][0], bounds[0][1], n)
    press = np.random.uniform(bounds[1][0], bounds[1][1], n)
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
            filein = "output/out-"+str(p)+".data"
        else:
            filein = "output/out-best.data"
        
        #--------------Getting Fp difference----------------
        
        fp_Diff = SOAP(filein)

        #------------------------------------

        scores.append(1/fp_Diff) 

    os.system('mv  output gen_{}'.format(gen))
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

#-------------Restarting section-----------------
restart = False

if restart:
    data = open("restart.dat","r").readlines()[-1]
    pop = eval(data.split("|")[0])
    sores = eval(data.split("|")[1])

else:
    # generate a random initial population: weights and bias of neural networks
    pop = [[random.gauss(0,1) for _ in range(hidden_layer+input_layer*hidden_layer+output_layer*hidden_layer)] for _ in range(n_pop)]
    # evaluate all candidates in the population: run neural networks and LAMMPS
    scores = evaluate(pop, -1, n_pop)


#----------------------------------------------

# select the best candidate 
idx = scores.index(max(scores))
best, best_eval = idx, scores[idx]
print(">-1, new best = %f" % (best_eval))
print()


with open("dumpfile.dat","a") as outfile:
    outfile.write("{} {} {}\n".format(-1,idx,best_eval))    #generation pop_id, best_score



for gen in range(n_iter): # maximum number of iterations

    # rank the scores 
    indices = [scores.index(x) for x in sorted(scores, reverse=True)]

    # select parents from the current population
    # n_best candidates are selected to generate new candidates
    selected = [selection(np.take(pop,indices,0)[:n_best], np.take(scores,indices,0)[:n_best]) for _ in range(n_pop)]
    
    # create the next generation
    children = list()
    for i in range(0, n_pop):
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
    
    with open("dumpfile.dat","a") as outfile:
        outfile.write("{} {} {}\n".format(gen,idx,best_eval))

    with open("restart.dat","a") as outfile2:
        outfile2.write("{} | {}\n".format(pop,scores))

# evaluate the best candidate
scores = evaluate([pop[idx]], n_iter, 1)
print("best = %f" % (scores[0]))

with open("dumpfile.dat","a") as outfile:
    outfile.write("{} {} {}\n".format(n_iter,idx,best_eval))


#################################################################################
# end of main code
#################################################################################
