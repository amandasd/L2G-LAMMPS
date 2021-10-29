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
# usage: run_l2g.py [-h] [-gpus NUMBER_OF_GPUS] [-gen NUMBER_OF_GENERATIONS] \ 
#        [-pop POPULATION_SIZE] [-mr MUTATION_RATE] [-ts TOURNAMENT_SIZE]    \
#        [-best NUMBER_OF_RETAINED_SOLUTIONS] [-elitism] [-hid NUMBER_OF_HIDDEN_NODES] [-restart] 
#
# optional arguments:
#  -h, --help                                                           show this help message and exit
#  -restart, --restart                                                  restart L2G from the last state in case it was interrupted [default=False]
#  -gpus,    --number-of-gpus NUMBER_OF_GPUS                            number of gpus [default=1]
#  -gen,     --number-of-generations NUMBER_OF_GENERATIONS              number of generations [default=2]
#  -pop,     --population-size POPULATION_SIZE                          population size [default=8]
#  -mr,      --mutation-rate MUTATION_RATE                              mutation rate (value between 0 and 1) [default=1]
#  -ts,      --tournament-size TOURNAMENT_SIZE                          tournament size [default=3]
#  -best,    --number-of-retained-solutions NUMBER_OF_BEST_SOLUTIONS    number of best candidates selected to generate new candidates [default=4]
#  -e,       --elitism                                                  elitism [default=True]
#  -hid,     --number-of-hidden-nodes HIDDEN_NODES                      number of hidden nodes [default=10]
#
# Example:
#    python run_l2g.py -gpus 8 -gen 2 -pop 8 -mr 1 -ts 3 -best 3 -e -hid 10
#################################################################################

#################################################################################
# parameters
#################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-gpus", "--number-of-gpus", type=int, default=1, help="number of gpus [default=1]")
parser.add_argument("-restart", "--restart", type=bool, default=False, help="restart L2G from the last state in case it was interrupted [default=False]")

# genetic algorithm parameters
parser.add_argument("-gen", "--number-of-generations", type=int, default=2, help="number of generations [default=2]")
parser.add_argument("-pop", "--population-size", type=int, default=8, help="population size [default=8]")
parser.add_argument("-mr", "--mutation-rate", type=float, default=1, help="mutation rate (value between 0 and 1) [default=1]")
parser.add_argument("-ts", "--tournament-size", type=int, default=3, help="tournament size [default=3]")
parser.add_argument("-best", "--number-of-retained-solutions", type=int, default=4, help="number of best candidates selected to generate new candidates [default=4]")
parser.add_argument("-e", "--elitism", type=bool, default=True, help="elitism [default=True]")

# neural network parameters 
parser.add_argument("-hid", "--number-of-hidden-nodes", type=int, default=10, help="number of hidden nodes [default=10]")
args = parser.parse_args()

# define genetic algorithm parameters
n_gen    = args.number_of_generations
n_pop    = args.population_size
mut_rate = args.mutation_rate
ts       = args.tournament_size
n_best   = args.number_of_retained_solutions 

if mut_rate > 1 or mut_rate < 0:
    mut_rate = 1
if n_best > n_pop:
    n_best = int(np.ceil(n_pop * 0.1))

# define neural network parameters 
input_nodes  = 1
hidden_nodes = args.number_of_hidden_nodes
output_nodes = 2

#################################################################################
# end of parameters
#################################################################################

#################################################################################
# functions
#################################################################################

# tournament selection
def selection(pop, scores, k=ts):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
    	    selection_ix = ix
    return pop[selection_ix]


# mutation operator
def mutation(ind, mu, sigma, mut_rate):
    for i in range(len(ind)):
        # check for a mutation
        if rand() < mut_rate:
            ind[i] += random.gauss(mu,sigma)
        else:
            ind[i] = random.gauss(0,1)
#TODO: do I need to check bounds?


# run neural networks
def run_networks(pop, temp, press, node_input, n):

    for p in range(n):

        # get weights and bias
        bias_hidden = pop[p][:hidden_nodes]
        weight_ih = pop[p][hidden_nodes:hidden_nodes+input_nodes*hidden_nodes]
        weight_ih = np.reshape(weight_ih,(input_nodes,hidden_nodes))
        weight_ho = pop[p][hidden_nodes+input_nodes*hidden_nodes:hidden_nodes+input_nodes*hidden_nodes+output_nodes*hidden_nodes]
        weight_ho = np.reshape(weight_ho,(hidden_nodes,output_nodes))
 
        # calculate Sj 
        node_hidden = np.zeros(shape=(hidden_nodes), dtype=float)
        for j in range(hidden_nodes):
            node_hidden[j] = np.sum(np.dot(node_input,weight_ih[:,j]))
        node_hidden = np.tanh(np.add(node_hidden,bias_hidden))
        
        # calculate Sk 
        node_output = np.zeros(shape=(output_nodes), dtype=float)
        for k in range(output_nodes):
            node_output[k] = np.sum(np.dot(node_hidden,weight_ho[:,k]))/(1.*hidden_nodes)

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
    if gen <= n_gen:
        lmp.run_lammps(temp, press, 0, gen, n_pop, args.number_of_gpus) 
    else:
        lmp.best_lammps(temp, press, 0, gen)

    for s in range(lmp.n_steps):
        node_input = s * 1./lmp.n_steps 
        # run neural networks
        temp, press = run_networks(pop, temp, press, node_input, n)
        state = s*lmp.npt_steps+lmp.npt_steps+lmp.nve_steps
        # run LAMMPS
        if gen <= n_gen:
            lmp.run_lammps(temp, press, state, gen, n_pop, args.number_of_gpus) 
        else:
            lmp.best_lammps(temp, press, state, gen)

    # calculate scores
    scores = lmp.get_scores(gen, n, n_gen)
    
    lmp.delete_output_files(gen, n, n_gen)

    return scores

#################################################################################
# end of functions
#################################################################################

#################################################################################
# learning to grow (L2G): main code
#################################################################################

print()
print("-gpus "+str(args.number_of_gpus)+" -gen "+str(n_gen)+" -pop "+str(n_pop)+" -mr "+str(mut_rate)+" -ts "+str(ts)+" -best "+str(n_best)+" -elitism "+str(args.elitism)+" -hid "+str(hidden_nodes)+" -restart "+str(args.restart))
print()

random.seed(datetime.now())

# restart L2G from the last state in case it was interrupted
if args.restart:
    value = -1000
    lines = open("restart.dat","r").readlines()
    for line_idx, data in enumerate(lines):
        gen_scores = eval(data.split("|")[1])    
        if max(gen_scores) > value:
            pop = eval(data.split("|")[0])
            scores = eval(data.split("|")[1])
            value = max(gen_scores)
            print('Gen:', line_idx, 'in previous run had better score. Selecting that...')        
else:
    # generate a random initial population: weights and bias of neural networks
    pop = [[random.gauss(0,1) for _ in range(hidden_nodes+input_nodes*hidden_nodes+output_nodes*hidden_nodes)] for _ in range(n_pop)]
    # evaluate all candidates in the population: run neural networks and LAMMPS
    scores = evaluate(pop, 0, n_pop)

# select the best candidate 
idx = scores.index(max(scores))
best_ind, best_score = pop[idx], scores[idx]
print(">0, new best = %f" % (best_score))
print()

for gen in range(n_gen): # maximum number of iterations

    # rank the scores 
    indices = [scores.index(x) for x in sorted(scores, reverse=True)]

    # select parents from the current population
    # n_best candidates are selected to generate new candidates
    selected = [selection(np.take(pop,indices,0)[:n_best], np.take(scores,indices,0)[:n_best]) for _ in range(n_pop)]
    
    # create the next generation
    new_pop = list()
    for i in range(0, n_pop):
        #copy the best candidate to next generation without mutation
        if args.elitism:
            new_pop.append(pop[idx])
            elitism = False
            continue
        ind = selected[i]
        # mutation: change weights and bias of neural networks
        mutation(ind, 0, 0.01, mut_rate)
        # store for next generation
        new_pop.append(ind)
    # replace population
    pop = new_pop

    # evaluate all candidates in the population: run neural networks and LAMMPS
    scores = evaluate(pop, gen+1, n_pop)

    # save population and scores in order to restart L2G from the last state in case of being interrupted
    with open("restart.dat","a") as outfile:
        outfile.write("{} | {}\n".format([p.tolist() for p in pop],scores))

    # select the best candidate
    idx = scores.index(max(scores))
    if scores[idx] > best_score:
        best_ind, best_score = pop[idx], scores[idx]
        print(">%d, new best = %f" % (gen+1, best_score))
        print()

# evaluate the best candidate
scores = evaluate([pop[idx]], n_gen+1, 1)
print("best = %f" % (scores[0]))

#################################################################################
# end of main code
#################################################################################
