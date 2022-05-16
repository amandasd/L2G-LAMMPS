#!/usr/bin/env python

from numpy.random import randint
from numpy.random import rand
import random

from datetime import datetime

import numpy as np
import os,sys
import argparse

import time

import module_lammps as lmp

#####################################################################################################
# Learning to Grow algorithm for LAMMPS (L2G)
# Solution is an array of two real values that represents temperature and pressure values
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
#  -gen,     --number-of-generations NUMBER_OF_GENERATIONS              number of generations [default=2]
#  -pop,     --population-size POPULATION_SIZE                          population size [default=8]
#  -popf,    --population-factor PRESSURE_FACTOR                        population factor [default=2]
#  -mr,      --mutation-rate MUTATION_RATE                              mutation rate (value between 0 and 1) [default=1]
#  -ms,      --mutation-sigma MUTATION_SIGMA                            sigma of gaussian random number (value between 0 and 1) [default=0.01]
#  -best,    --number-of-retained-solutions NUMBER_OF_BEST_SOLUTIONS    number of best candidates selected to generate new candidates [default=4]
#  -e,       --elitism                                                  elitism [default=True]
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
parser.add_argument("-r", "--restart", type=bool, default=False, help="restart L2G from the last state in case it was interrupted [default=False]")

# genetic algorithm parameters
parser.add_argument("-gen", "--number-of-generations", type=int, default=2, help="number of generations [default=2]")
parser.add_argument("-pop", "--population-size", type=int, default=8, help="population size [default=8]")
parser.add_argument("-popf", "--population-factor", type=int, default=2, help="population factor [default=2]")
parser.add_argument("-mr", "--mutation-rate", type=float, default=1, help="mutation rate (value between 0 and 1) [default=1]")
parser.add_argument("-ms", "--mutation-sigma", type=float, default=0.01, help="sigma of gaussian random number (value between 0 and 1) [default=0.01]")
parser.add_argument("-best", "--number-of-retained-solutions", type=int, default=4, help="number of best candidates selected to generate new candidates [default=4]")
parser.add_argument("-e", "--elitism", type=bool, default=True, help="elitism [default=True]")

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
mut_rate  = args.mutation_rate
mut_sigma = args.mutation_sigma
n_best    = args.number_of_retained_solutions

if mut_rate > 1 or mut_rate < 0:
    mut_rate = 1
if n_best > n_pop:
    n_best = int(np.ceil(n_pop * 0.1))

#define temperature and pressure parameters
bounds = [[args.minimum_temperature, args.maximum_temperature], [args.minimum_pressure, args.maximum_pressure]]

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
#TODO: do I need to check bounds?


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


# evaluate all candidates in the population: run neural networks and LAMMPS
def evaluate(pop, gen, n, temp, press):

    # run LAMMPS with initial structure
    if gen <= n_gen:
        print("[gen %s] running LAMMPS with initial structure" %str(gen))
        start = time.perf_counter()
        lmp.run_lammps(temp, press, 0, gen, n, args.number_of_gpus)
        end = time.perf_counter()
        #print(end-start)
    else:
        print("running LAMMPS with initial structure for best solution")
        lmp.best_lammps(temp, press, 0, gen)

    for s in range(lmp.n_steps):
        state = s*lmp.npt_steps+lmp.npt_steps+lmp.nve_steps
        # run LAMMPS with restart files
        if gen <= n_gen:
            print("[gen %s; step %s] running LAMMPS with restart file" %(str(gen), str(s)))
            start = time.perf_counter()
            lmp.run_lammps(temp, press, state, gen, n, args.number_of_gpus)
            end = time.perf_counter()
            #print(end-start)
        else:
            print("[step %s] running LAMMPS with restart file for best solution" %(str(s)))
            lmp.best_lammps(temp, press, state, gen)

    # calculate scores
    scores = lmp.get_scores(gen, n, n_gen)

    lmp.delete_output_files(gen, n, n_gen)

    return scores

#################################################################################
# end of functions
#################################################################################

#################################################################################
# Learning to Grow for LAMMPS: main code
#################################################################################
if __name__ == '__main__':

    print()
    print("-gpus "+str(args.number_of_gpus)+" -gen "+str(n_gen)+" -pop "+str(n_pop)+" -popf "+str(args.population_factor)+" -mr "+str(mut_rate)+" -ms "+str(mut_sigma)+" -best "+str(n_best)+" -elitism "+str(args.elitism)+" -restart "+str(args.restart)+" -tmin "+str(args.minimum_temperature)+" -tmax "+str(args.maximum_temperature)+" -pmin "+str(args.minimum_pressure)+" -pmax "+str(args.maximum_pressure)+" -opt "+str(args.initialize_T_P)+" -vtemp "+str(args.initial_temperature)+" -vpress "+str(args.initial_pressure)+" -tf "+str(args.temperature_factor)+" -pf "+str(args.pressure_factor))
    print()

    # delete previous files: restart.dat and dumpfile.dat
    os.system('rm -f output/restart.dat')
    os.system('rm -f output/dumpfile.dat')
    os.system('rm -f output/protocol*')

    random.seed(datetime.now())

    temp, press = initialize_T_P(n_pop, args.initialize_T_P, args.initial_temperature, args.initial_pressure)

    # generate a random initial population
    pop = [[random.gauss(0,1) for _ in range(2)] for _ in range(n_pop*args.population_factor)]
    # evaluate all candidates in the population: run LAMMPS
    scores = evaluate(pop, 0, n_pop*args.population_factor, temp, press)

    # select the best candidate
    idx = scores.index(max(scores))
    #best_ind, best_score = pop[idx], scores[idx]
    #print()
    #print(">0, new best = %f" % (best_score))
    #print()

    # save generation, best index, and best score in an output file
    with open("output/dumpfile.dat","a") as outfile1:
        outfile1.write("{} {} {}\n".format(0, idx, scores[idx]))

    for gen in range(n_gen): # maximum number of iterations

        elitism = args.elitism

        # rank the scores
        indices = [scores.index(x) for x in sorted(scores, reverse=True)]

        if gen == 0:
            selected_idx = list(np.array(indices)[np.array(randint(0, n_best, n_pop*args.population_factor))])
        else:
            selected_idx = list(np.array(indices)[np.array(randint(0, n_best, n_pop))])
        selected = list(np.array(pop)[np.array(selected_idx)])

        if gen == 0:
            for i in range(0, n_pop):
                with open("output/protocol-"+str(i),"a") as outfile:
                    outfile.write("gen {}, {}, {}\n".format(gen,temp[i],press[i]))

        # create the next generation
        new_pop = list()
        for i in range(0, n_pop):

            #copy the best candidate to next generation without mutation
            if elitism:
                new_pop.append(pop[idx])
                temp[i] = temp[idx]
                press[i] = press[idx]
                elitism = False
                with open("output/protocol-"+str(i),"a") as outfile:
                    outfile.write("gen {}, {}, {}\n".format(gen+1,temp[i],press[i]))
                continue

            ind = selected[i]
            mutation(ind, 0, mut_sigma, mut_rate)

            temp[i] = temp[selected_idx[i]] + ind[0] * args.temperature_factor * random.choice([1,-1])
            if temp[i] > bounds[0][1]:
                temp[i] = bounds[0][1]
            if temp[i] < bounds[0][0]:
                temp[i] = bounds[0][0]

            press[i] = press[selected_idx[i]] + ind[1] * args.pressure_factor * random.choice([1,-1])
            if press[i] > bounds[1][1]:
                press[i] = bounds[1][1]
            if press[i] < bounds[1][0]:
                press[i] = bounds[1][0]

            with open("output/protocol-"+str(i),"a") as outfile:
                outfile.write("gen {}, {}, {}\n".format(gen+1,temp[i],press[i]))

           # store for next generation
            new_pop.append(ind)

        # replace population
        pop = new_pop

        # evaluate all candidates in the population: run neural networks and LAMMPS
        scores = evaluate(pop, gen+1, n_pop, temp, press)

        # save population and scores in order to restart L2G from the last state in case of being interrupted
        with open("output/restart.dat","a") as outfile2:
            outfile2.write("{} | {}\n".format(pop,scores))

        # select the best candidate
        idx = scores.index(max(scores))
        #if scores[idx] > best_score:
        #    best_ind, best_score = pop[idx], scores[idx]
        #    print()
        #    print(">%d, new best = %f" % (gen+1, best_score))
        #    print()

        # save generation, best index, and best score in an output file
        with open("output/dumpfile.dat","a") as outfile1:
            outfile1.write("{} {} {}\n".format(gen+1, idx, scores[idx]))

    # evaluate the best candidate
    #scores = evaluate([pop[idx]], n_gen+1, 1)
    #print("best = %f" % (scores[0]))

#################################################################################
# end of main code
#################################################################################
