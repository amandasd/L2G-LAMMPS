#!/usr/bin/env python

from numpy.random import randint

import numpy as np
import os,sys

from ovito.modifiers import PolyhedralTemplateMatchingModifier
from ovito.io import import_file

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
        # 0 < seed <= 8 digits
        newdata = newdata.replace("variable                seed equal 1","variable                seed equal {}".format(randint(0, 99999999)))
        newdata = newdata.replace("restart         ${npt_steps} He.restart","restart         {} output/He-{}.restart".format(npt_steps,p))

        if state > 0:
            newdata = newdata.replace("read_restart    input/He.restart.1000000","read_restart    output/He-"+str(p)+".restart."+str(state))
            newdata = newdata.replace("dump            1 all custom 10000 output/He.xyz id type x y z","dump            1 all custom {} output/He-{}-{}-{}.xyz id type x y z".format(dump_freq,gen,p,(state-nve_steps)/npt_steps))
        else:
            newdata = newdata.replace("dump            1 all custom 10000 output/He.xyz id type x y z","dump            1 all custom {} output/He-{}-{}-{}.xyz id type x y z".format(dump_freq,gen,p,state))

        fileout = "input/in."+str(p)
        f = open(fileout,'w')
        f.write(newdata)
        f.close()

    # run LAMMPS for each candidate in the population
    if state > 0:
        os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(n_gpus)+' 0'+' '+str(gen)+' '+str(int((state-nve_steps)/npt_steps)))
    else:
        os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(n_gpus)+' 0'+' '+str(gen)+' '+str(state))


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
    os.system('./scripts/run_lammps.sh 1 1 1 -1 -1')


def get_scores(n, ref_phase='BCC', weights = [1], frame_id=None):
    ks = ['PolyhedralTemplateMatching.counts.%s'%ref_phase,]
    scores = []
    for p in range(n):
        filein = "output/He-"+str(p)+".xyz"

        pipeline = import_file(filein)
        modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1, only_selected=False)
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.OTHER].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.BCC].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = False
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = False
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = False
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = False
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = False
        pipeline.modifiers.append(modifier)

        if frame_id is not None:
            print('Computing for user-specified frame # %s'%frame_id)
            data = pipeline.compute(frame_id)
        elif pipeline.source.num_frames > 1:
            print('Multiple frames found, computing for last frame')
            data = pipeline.compute(pipeline.source.num_frames)
        else:
            data = pipeline.compute()

        num_particles = data.particles.count

        sc = 0
        for idx,k in enumerate(ks):
            sc += weights[idx] * data.attributes[k]/num_particles

        scores.append(sc)

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
        os.system('rm -f output/He-*.xyz')
        #os.system('rm -f output/out.*')

#################################################################################
# end of functions
#################################################################################
