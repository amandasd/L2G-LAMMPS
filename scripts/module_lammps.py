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
dump_freq = 60000

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
            newdata = newdata.replace("dump            1 all custom 10000 output/He.xyz id type x y z c_qlwlhat[1] c_qlwlhat[2] c_qlwlhat[3] c_qlwlhat[4] c_qlwlhat[5]","dump            1 all custom {} output/He-{}-{}-{}.xyz id type x y z c_qlwlhat[1] c_qlwlhat[2] c_qlwlhat[3] c_qlwlhat[4] c_qlwlhat[5]".format(dump_freq,gen,p,int((state-nve_steps)/npt_steps)))
        else:
            newdata = newdata.replace("dump            1 all custom 10000 output/He.xyz id type x y z c_qlwlhat[1] c_qlwlhat[2] c_qlwlhat[3] c_qlwlhat[4] c_qlwlhat[5]","dump            1 all custom {} output/He-{}-{}-{}.xyz id type x y z c_qlwlhat[1] c_qlwlhat[2] c_qlwlhat[3] c_qlwlhat[4] c_qlwlhat[5]".format(dump_freq,gen,p,state))

        fileout = "input/in."+str(p)
        f = open(fileout,'w')
        f.write(newdata)
        f.close()

    # run LAMMPS for each candidate in the population
    if state > 0:
        os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(n_gpus)+' 0'+' '+str(gen)+' '+str(int((state-nve_steps)/npt_steps)))
    else:
        os.system('./scripts/run_lammps.sh '+str(n_pop)+' '+str(n_gpus)+' 0'+' '+str(gen)+' '+str(state))


def get_scores(gen, n, state, ref_phase='FCC', weights = [1], frame_id=None):

    if state > 0:
        step = (int((state-nve_steps)/npt_steps))
    else:
        step = state

    scores = []
    for p in range(n):
        filein = "output/He-"+str(gen)+"-"+str(p)+"-"+str(step)+".xyz"
        f = open(filein,'r')
        lines = f.readlines()
        f.close()

        pipeline = import_file(filein)
        modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1, only_selected=False)
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.OTHER].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.FCC].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.HCP].enabled = False
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

        ks = ['PolyhedralTemplateMatching.counts.FCC']
        for idx,k in enumerate(ks):
            fcc = weights[idx] * data.attributes[k]/num_particles

        ks = ['PolyhedralTemplateMatching.counts.BCC']
        for idx,k in enumerate(ks):
            bcc = weights[idx] * data.attributes[k]/num_particles

        ks = ['PolyhedralTemplateMatching.counts.OTHER']
        for idx,k in enumerate(ks):
            other = weights[idx] * data.attributes[k]/num_particles

        for i in range(len(lines)):
            #if 'ITEM: NUMBER OF ATOMS' in lines[i]:
            #   num_particles = int(lines[i+1][0])
            #   continue
            if 'ITEM: ATOMS id type x y z c_qlwlhat[1] c_qlwlhat[2] c_qlwlhat[3] c_qlwlhat[4] c_qlwlhat[5]' in lines[i]:
               line = i+1
               break
        q = 0
        for i in range(line,len(lines)):
            q4 = float(lines[i].split(' ')[5])
            q6 = float(lines[i].split(' ')[6])
            q8 = float(lines[i].split(' ')[7])
            q10 = float(lines[i].split(' ')[8])
            q12 = float(lines[i].split(' ')[9])
            if ref_phase == "FCC":
               q += ((q4-0.19094063)**2) + ((q6-0.5745243)**2) + ((q8-0.4039145)**2) + ((q10-0.01285705)**2) + ((q12-0.600083)**2)
            elif ref_phase == "BCC":
               q += ((q4-0.22402525)**2) + ((q6-0.56693995)**2) + ((q8-0.32595545)**2) + ((q10-0.41235015)**2) + ((q12-0.3760503)**2)

        with open("output/score-"+str(gen)+"-"+str(p)+"."+str(step),"a") as outfile:
            outfile.write("{}, {}, {}, {}\n".format(fcc, bcc, other, q/(4.0*num_particles)))

        if step == n_steps:
            scores.append(q/(4.0*num_particles))

    return scores


def delete_output_files(gen, n, n_iter):
    ini_state = npt_steps+nve_steps
    end_state = n_steps*npt_steps+npt_steps+nve_steps+nve_steps
    # delete restart files
    for i in np.arange(ini_state,end_state,1000):
        if gen <= n_iter:
            os.system('rm -f output/He-*.restart.'+str(i))
    # delete dump and out files
    if gen <= n_iter:
        os.system('rm -f output/He-*.dump')
        #os.system('rm -f output/out.*')

    nfile = 'output/dumpfile.dat'
    try:
       f = open(nfile,"r")
    except IOError:
       print >> sys.stderr, "Could not open file " + nfile
       sys.exit(1)
    dump_lines = f.readlines()
    f.close()

    ind = int(dump_lines[gen].split(' ')[1])
    os.system('bash -c \"rm output/He-'+str(gen)+'-!('+str(ind)+')-*.xyz\"')


#################################################################################
# end of functions
#################################################################################
