#!/usr/bin/env python

from numpy.random import randint
from numpy.random import rand
import random

from datetime import datetime

import numpy as np
import os,sys
import argparse

import module_lammps as lmp

#####################################################################################################
# Learning to Grow algorithm for LAMMPS (L2G)
# Solution is an array of real values that represents a neural network
# Changes must be made to the module_lammps.py file according to your LAMMPS simulation
# References
# 1 - S. Whitelam, I. Tamblyn. "Learning to grow: control of materials self-assembly using
#     evolutionary reinforcement learning". Phys. Rev. E, 2020. DOI: 10.1103/PhysRevE.101.052604
# 2 - https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
#####################################################################################################
# usage: run_l2g.py [-h] [-gpus NUMBER_OF_GPUS] [-gen NUMBER_OF_GENERATIONS] \
#        [-pop POPULATION_SIZE] [-popf POPULATION_FACTOR] [-mr MUTATION_RATE] [-ms MUTATION_SIGMA] [-ts TOURNAMENT_SIZE]    \
#        [-best NUMBER_OF_RETAINED_SOLUTIONS] [-elitism] [-hid NUMBER_OF_HIDDEN_NODES] [-restart] \
#        [-tmin MINIMUM_TEMPERATURE] [-tmax MAXIMUM_TEMPERATURE] [-pmin MINIMUM_PRESSURE] \
#        [-pmax MAXIMUM_PRESSURE] [-opt OPT] [-vtemp INITIAL_TEMPERATURE] [-vpress INITIAL_PRESSURE] \
#        [-tf TEMPERATURE_FACTOR] [-pf PRESSURE_FACTOR]
#
# arguments:
#  -h, --help                                                           show this help message and exit
#  -r, --restart                                                        restart L2G from the last state in case it was interrupted [default=False]
#  -gpus,    --number-of-gpus NUMBER_OF_GPUS                            number of gpus [default=1]
#  -s,       --strategy STRATEGY                                        strategy: ga, GA, mc, MC [default=ga]
#  -gen,     --number-of-generations NUMBER_OF_GENERATIONS              number of generations [default=2]
#  -pop,     --population-size POPULATION_SIZE                          population size [default=1]
#  -popf,    --population-factor PRESSURE_FACTOR                        population factor [default=1]
#  -mr,      --mutation-rate MUTATION_RATE                              mutation rate (value between 0 and 1) [default=1]
#  -ms,      --mutation-sigma MUTATION_SIGMA                            sigma of gaussian random number (value between 0 and 1) [default=0.01]
#  -best,    --number-of-retained-solutions NUMBER_OF_BEST_SOLUTIONS    number of best candidates selected to generate new candidates [default=1]
#  -e,       --elitism                                                  elitism [default=False]
#  -sim,     --number-of-simulations NUMBER_OF_SIMULATIONS              number of simulations for each candidate solution [default=1]
#  -hid,     --number-of-hidden-nodes HIDDEN_NODES                      number of hidden nodes [default=10]
#  -input,   --number-of-input-nodes INPUT_NODES                        number of input nodes: 1, 3 [default=1]
#  -tmin,    --minimum-temperature MINIMUM_TEMPERATURE                  minimum temperature value [default=0.5]
#  -tmax,    --maximum-temperature MAXIMUM_TEMPERATURE                  maximum temperature value [default=2]
#  -pmin,    --minimum-pressure MINIMUM_PRESSURE                        minimum pressure value [default=0.5]
#  -pmax,    --maximim-pressure MAXIMUM_PRESSURE                        maximum pressure value [default=1]
#  -opt",    --initialize-T-P OPTION                                    valid options to initialize temperature and pressure values: 0 (random), 1 (fixed values), 2 (mutated from a given value) [default=0]. For options 1 and 2, -vtemp and -vpress are required
#  -vtemp,   --initial-temperature INITIAL_TEMPERATURE                  initial temperature value, a required argument for options 1 and 2 of initialize_T_P() function [default=None]
#  -vpress,  --initial-pressure INITIAL_PRESSURE                        initial pressure value, a required argument for options 1 and 2 of initialize_T_P() function [default=None]
#  -tf,      --temperature-factor TEMPERATURE_FACTOR                    temperature factor [default=10]
#  -pf,      --pressure-factor PRESSURE_FACTOR                          pressure factor [default=1000]
#
# Example:
#    python run_l2g.py -gpus 8 -gen 2 -pop 8 -mr 1 -ts 3 -best 3 -e -hid 10
#####################################################################################################

#################################################################################
# parameters
#################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-gpus", "--number-of-gpus", type=int, default=1, help="number of gpus [default=1]")
parser.add_argument("-s", "--strategy", choices=['ga', 'GA', 'mc', 'MC'], default="ga", help="Strategy: ga, GA, mc, MC [default=ga]")
parser.add_argument("-r", "--restart", type=bool, default=False, help="restart L2G from the last state in case it was interrupted [default=False]")

# genetic algorithm parameters
parser.add_argument("-gen", "--number-of-generations", type=int, default=2, help="number of generations [default=2]")
parser.add_argument("-pop", "--population-size", type=int, default=1, help="population size [default=1]")
parser.add_argument("-popf", "--population-factor", type=int, default=1, help="population factor [default=1]")
parser.add_argument("-mr", "--mutation-rate", type=float, default=1, help="mutation rate (value between 0 and 1) [default=1]")
parser.add_argument("-ms", "--mutation-sigma", type=float, default=0.01, help="sigma of gaussian random number (value between 0 and 1) [default=0.01]")
parser.add_argument("-best", "--number-of-retained-solutions", type=int, default=1, help="number of best candidates selected to generate new candidates [default=1]")
parser.add_argument("-e", "--elitism", type=bool, default=True, help="elitism [default=False]")

# monte carlo parameters
parser.add_argument("-sim", "--number-of-simulations", type=int, default=1, help="number of simulations for each candidate solution [default=1]")

# neural network parameters
parser.add_argument("-hid", "--number-of-hidden-nodes", type=int, default=10, help="number of hidden nodes [default=10]")
parser.add_argument("-input", "--number-of-input-nodes", type=int, choices=[1, 3], default=1, help="number of input nodes: 1, 3 [default=1]")

# temperature and pressure parameters
parser.add_argument("-tmin", "--minimum-temperature", type=float, default=0.5, help="minimum temperature value [default=0.5]")
parser.add_argument("-tmax", "--maximum-temperature", type=float, default=2, help="maximum temperature value [default=2]")
parser.add_argument("-pmin", "--minimum-pressure", type=float, default=0.5, help="minimum pressure value [default=0.5]")
parser.add_argument("-pmax", "--maximum-pressure", type=float, default=1, help="maximum pressure value [default=1]")
parser.add_argument("-opt", "--initialize-T-P", type=int, choices=[0, 1, 2], default=0, help="valid options to initialize temperature and pressure values: 0 (random), 1 (fixed values), 2 (mutated from a given value) [default=0]. For options 1 and 2, -vtemp and -vpress are required")
parser.add_argument("-vtemp", "--initial-temperature", type=float, default=None, help="initial temperature value, a required argument for options 1 and 2 of initialize_T_P() function [default=None]")
parser.add_argument("-vpress", "--initial-pressure", type=float, default=None, help="initial pressure value, a required argument for options 1 and 2 of initialize_T_P() function [default=None]")
parser.add_argument("-tf", "--temperature-factor", type=int, default=10, help="temperature factor [default=10]")
parser.add_argument("-pf", "--pressure-factor", type=int, default=1000, help="pressure factor [default=1000]")

args = parser.parse_args()

# define genetic algorithm parameters
n_gen     = args.number_of_generations
n_pop     = args.population_size
n_best    = args.number_of_retained_solutions
pop_f     = args.population_factor
mut_rate  = args.mutation_rate
mut_sigma = args.mutation_sigma
e_flag    = args.elitism

# define monte carlo parameter
n_sim = args.number_of_simulations

if mut_rate > 1 or mut_rate < 0:
    mut_rate = 1
if n_best > n_pop:
    n_best = int(np.ceil(n_pop * 0.1))

# define neural network parameters
input_nodes  = args.number_of_input_nodes
hidden_nodes = args.number_of_hidden_nodes
output_nodes = 2

#define temperature and pressure parameters
bounds = [[args.minimum_temperature, args.maximum_temperature], [args.minimum_pressure, args.maximum_pressure]]

if args.strategy.upper() == 'GA':
   n_sim  = 1
elif args.strategy.upper() == 'MC':
   n_pop  = 1
   n_best = 1
   pop_f  = 1
   e_flag = False


#################################################################################
# end of parameters
#################################################################################

#################################################################################
# functions
#################################################################################

# mutation operator
def mutation(ind, mu, sigma, mut_rate):
    for i in range(len(ind)):
        # check for a mutation
        if rand() < mut_rate:
            ind[i] += random.gauss(mu,sigma)
        else:
            ind[i] = random.gauss(0,1)


# initialize temperature and pressure values: 0 (random), 1 (fixed values), 2 (mutated from a given value)
def initialize_T_P(n, opt, vtemp=None, vpress=None):
    if opt == 0:
        temp  = np.random.uniform(bounds[0][0], bounds[0][1], n)
        press = np.random.uniform(bounds[1][0], bounds[1][1], n)
    elif opt == 1:
        if vtemp == None or vpress == None:
            print("usage: [-vtemp INITIAL_TEMPERATURE] [-vpress INITIAL_PRESSURE]")
            exit()
        temp  = np.full(n, vtemp, dtype=float)
        press = np.full(n, vpress, dtype=float)
    elif opt == 2:
        if vtemp == None or vpress == None:
            print("usage: [-vtemp INITIAL_TEMPERATURE] [-vpress INITIAL_PRESSURE]")
            exit()
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
        print("Valid options: 0 (random), 1 (fixed values), 2 (mutated from a given value)")

    return temp, press


# run neural networks
def run_networks(pop, temp, press, time, n, gen):

    for p in range(n):

        if int(time*lmp.n_steps) == 0:
            with open("output/protocol-"+str(p),"a") as outfile:
                outfile.write("gen {}, step {}, {}, {}\n".format(gen,int(time*lmp.n_steps),temp[p],press[p]))

        # get weights and bias
        bias_hidden = pop[p][:hidden_nodes]
        weight_ih = pop[p][hidden_nodes:hidden_nodes+input_nodes*hidden_nodes]
        weight_ih = np.reshape(weight_ih,(input_nodes,hidden_nodes))
        weight_ho = pop[p][hidden_nodes+input_nodes*hidden_nodes:hidden_nodes+input_nodes*hidden_nodes+output_nodes*hidden_nodes]
        weight_ho = np.reshape(weight_ho,(hidden_nodes,output_nodes))

        node_input = np.zeros(shape=(input_nodes), dtype=float)
        if input_nodes == 1:
            node_input[0] = time
        elif input_nodes == 3:
            node_input[0] = time
            node_input[1] = temp[p]
            node_input[2] = press[p]
        else:
            print("Valid options: 1 and 3")

        # calculate Sj
        node_hidden = np.zeros(shape=(hidden_nodes), dtype=float)
        for j in range(hidden_nodes):
            node_hidden[j] = np.sum(np.dot(node_input,weight_ih[:,j]))
        node_hidden = np.tanh(np.add(node_hidden,bias_hidden))

        # calculate Sk
        node_output = np.zeros(shape=(output_nodes), dtype=float)
        for k in range(output_nodes):
            node_output[k] = np.sum(np.dot(node_hidden,weight_ho[:,k]))/(1.*hidden_nodes)

        temp[p] += node_output[0] * args.temperature_factor
        if temp[p] < 0:
            temp[p] = 0

        press[p] += node_output[1] * args.pressure_factor
        if press[p] < 0:
            press[p] = 0

        with open("output/protocol-"+str(p),"a") as outfile:
            outfile.write("gen {}, step {}, {}, {}, {}, {}\n".format(gen,int(time*lmp.n_steps),temp[p],press[p],node_output[0],node_output[1]))

    return temp, press


# evaluate all candidates in the population: run neural networks and LAMMPS
def evaluate(pop, gen, n):

    # initialize temperature and pressure values: 0 (random), 1 (fixed values), 2 (mutated from a given value)
    # options 1 and 2 require initial temperature and pressure values
    # arguments: population size, option (0, 1, 2), initial temperature value, initial pressure value
    temp, press = initialize_T_P(n, args.initialize_T_P, args.initial_temperature, args.initial_pressure)

    # run LAMMPS with initial structure
    if gen <= n_gen:
        print("[gen %s] running LAMMPS with initial structure" %str(gen))
        lmp.run_lammps(temp, press, 0, gen, n, args.number_of_gpus)

        # save partial scores
        lmp.get_scores(gen, n, 0)

    for s in range(lmp.n_steps):
        time = s * 1./lmp.n_steps
        # run neural networks
        temp, press = run_networks(pop, temp, press, time, n, gen)
        state = s*lmp.npt_steps+lmp.npt_steps+lmp.nve_steps

        # run LAMMPS with restart files
        if gen <= n_gen:
            print("[gen %s; step %s] running LAMMPS with restart file" %(str(gen), str(s)))
            lmp.run_lammps(temp, press, state, gen, n, args.number_of_gpus)

        if s+1 == lmp.n_steps:
            # calculate scores
            scores, particles = lmp.get_scores(gen, n, state)
        else:
            # save partial scores
            lmp.get_scores(gen, n, state)

    #lmp.delete_output_files(gen, n, n_gen)

    return scores, particles

#################################################################################
# end of functions
#################################################################################

#################################################################################
# Learning to Grow for LAMMPS: main code
#################################################################################
if __name__ == '__main__':

    print()
    print("-s "+str(args.strategy)+" -gpus "+str(args.number_of_gpus)+" -gen "+str(n_gen)+" -pop "+str(n_pop)+" -popf "+str(pop_f)+" -sim "+str(n_sim)+" -mr "+str(mut_rate)+" -ms "+str(mut_sigma)+" -best "+str(n_best)+" -elitism "+str(e_flag)+" -hid "+str(hidden_nodes)+" -input "+str(input_nodes)+" -restart "+str(args.restart)+" -tmin "+str(args.minimum_temperature)+" -tmax "+str(args.maximum_temperature)+" -pmin "+str(args.minimum_pressure)+" -pmax "+str(args.maximum_pressure)+" -opt "+str(args.initialize_T_P)+" -vtemp "+str(args.initial_temperature)+" -vpress "+str(args.initial_pressure)+" -tf "+str(args.temperature_factor)+" -pf "+str(args.pressure_factor))
    print()

    # delete previous files
    os.system('rm -f output/restart.dat')
    os.system('rm -f output/dumpfile.dat')
    os.system('rm -f output/protocol*')
    os.system('rm -f output/He-*.xyz')
    os.system('rm -f output/score*')

    random.seed(datetime.now().timestamp())

    # generate a random initial population: weights and bias of neural networks
    pop = [[random.gauss(0,1) for _ in range(hidden_nodes+input_nodes*hidden_nodes+output_nodes*hidden_nodes)] for _ in range(n_pop*pop_f)]
    # n-sim-plicate the member of pop
    pop = (np.repeat(pop, repeats=n_sim, axis=0)).tolist()

    # evaluate all candidates in the population: run neural networks and LAMMPS
    scores, particles = evaluate(pop, 0, len(pop))

    if args.strategy.upper() == 'GA':
        # select the best candidate
        idx = scores.index(min(scores))
    elif args.strategy.upper() == 'MC':
        # average scores
        idx = 0; scores[idx] = np.mean(scores)
        particles[idx] = np.mean(particles)

    # save generation, best index, and best score in an output file
    with open("output/dumpfile.dat","a") as outfile1:
        outfile1.write("{} {} {}\n".format(0, idx, scores[idx]))

    lmp.delete_output_files(0, len(pop), n_gen)

    for gen in range(n_gen): # maximum number of iterations

        elitism = e_flag

        # rank the scores for minimum problem
        indices = list(np.argsort(scores))
        # rank the scores for maximum problem
        #indices = list(np.argsort(scores)[::-1])

        # select parents from the current population
        # n_best candidates are selected to generate new candidates
        selected_idx = list(np.array(indices)[np.array(randint(0, n_best, len(pop)))])
        selected = list(np.array(pop)[np.array(selected_idx)])

        # create the next generation
        new_pop = list()
        for i in range(0, n_pop):
            #copy the best candidate to next generation without mutation
            if elitism:
                new_pop.append(pop[idx])
                elitism = False
                continue
            ind = selected[i]
            # mutation: change weights and bias of neural networks
            mutation(ind, 0, mut_sigma, mut_rate)
            # store for next generation
            new_pop.append(ind)

        if args.strategy.upper() == 'GA':
            # replace population
            pop = new_pop

            # evaluate all candidates in the population: run neural networks and LAMMPS
            scores, particles = evaluate(pop, gen+1, len(pop))

            # select the best candidate
            idx = scores.index(min(scores))
        elif args.strategy.upper() == 'MC':
            # n-sim-plicate the member of pop
            new_pop = (np.repeat(new_pop, repeats=n_sim, axis=0)).tolist()

            # evaluate all candidates in the population: run neural networks and LAMMPS
            new_scores, new_particles = evaluate(new_pop, gen+1, len(new_pop))

            # average scores
            idx = 0; new_scores[idx] = np.mean(new_scores)
            new_particles[idx] = np.mean(new_particles)

            # minimum problem
            if (new_scores[idx] < scores[idx]) || new_particles[idx] > particles[idx]:
                pop = new_pop
                scores = new_scores

        # save population and scores in order to restart L2G from the last state in case of being interrupted
        with open("output/restart.dat","a") as outfile2:
            outfile2.write("{} | {}\n".format(pop,scores))

        # save generation, best index, and best score in an output file
        with open("output/dumpfile.dat","a") as outfile1:
            outfile1.write("{} {} {}\n".format(gen+1, idx, scores[idx]))

        lmp.delete_output_files(gen+1, len(pop), n_gen)

#################################################################################
# end of main code
#################################################################################
