#!/usr/bin/env python

from numpy.random import randint
from numpy.random import rand
import random

from datetime import datetime 

import numpy as np
import os,sys
import argparse

import module_lammps as lmp

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
args = parser.parse_args()

#################################################################################
# parameters
#################################################################################

# define genetic algorithm parameters
n_iter = 2 # define the total iterations
n_pop = 8 # define the population size
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
        if temp[p] > lmp.bounds[0][1]:
            temp[p] = lmp.bounds[0][1]
        if temp[p] < lmp.bounds[0][0]:
            temp[p] = lmp.bounds[0][0]
        
        press[p] += node_output[1]
        if press[p] > lmp.bounds[1][1]:
            press[p] = lmp.bounds[1][1]
        if press[p] < lmp.bounds[1][0]:
            press[p] = lmp.bounds[1][0]

    return temp, press


# evaluate all candidates in the population: run neural networks and LAMMPS
def evaluate(pop, gen, n):

    # initialize temperature and pressure values: 0 (random), 1 (fixed values), 2 (mutated from a given value)
    temp, press = lmp.initialize_T_P(n, 1, 1, 0.7) 

    # run LAMMPS
    if gen <= n_iter:
        lmp.run_lammps(temp, press, 0, gen, n_pop, args.number_of_gpus) 
    else:
        lmp.best_lammps(temp, press, 0, gen)

    for s in range(lmp.n_steps):
        node_input = s * 1./lmp.n_steps 
        # run neural networks
        temp, press = run_networks(pop, temp, press, node_input, n)
        state = s*lmp.npt_steps+lmp.npt_steps+lmp.nve_steps
        # run LAMMPS
        if gen <= n_iter:
            lmp.run_lammps(temp, press, state, gen, n_pop, args.number_of_gpus) 
        else:
            lmp.best_lammps(temp, press, state, gen)

    # calculate scores
    scores = lmp.get_scores(gen, n, n_iter)
    
    lmp.delete_output_files(state, gen, n, n_iter)

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
scores = evaluate(pop, 0, n_pop)

# select the best candidate 
idx = scores.index(max(scores))
best, best_eval = pop[idx], scores[idx]
print(">0, new best = %f" % (best_eval))
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
    scores = evaluate(pop, gen+1, n_pop)

    # select the best candidate
    idx = scores.index(max(scores))
    if scores[idx] > best_eval:
        best, best_eval = pop[idx], scores[idx]
        print(">%d, new best = %f" % (gen+1, best_eval))
        print()

# evaluate the best candidate
scores = evaluate([pop[idx]], n_iter+1, 1)
print("best = %f" % (scores[0]))

#################################################################################
# end of main code
#################################################################################
