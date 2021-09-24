import os
import pandas as pd
import numpy as np
from glob import glob
from pymatgen.io.vasp import Poscar
from pymatgen import Structure,Lattice
from dscribe.descriptors import MBTR,SOAP
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
from pymatgen import Structure,Lattice



#---------------------------------
def StructureToAse(structure):
    '''
    Used to create an ASE object from the structure dictionary of MCTS. 
    '''
    ASEobject = AseAtomsAdaptor().get_atoms(structure)
    ASEobject.set_cell([structure.lattice.a,structure.lattice.b,structure.lattice.c,structure.lattice.alpha,
                        structure.lattice.beta,structure.lattice.gamma])
    
    return ASEobject

#----------------------------------



def get_species(structure):
    '''
    Returns all the species as a list
    '''
    species = [cord.specie.symbol for cord in structure.sites]
        
    return species

#---------------------------------

def Get_SOAP(structure):
    '''
    For creation of SOAP Fingerpeinting
    '''
    ASEobject = StructureToAse(structure)
    species = list(set(get_species(structure)))
    rcut = 3
    nmax = 3
    lmax = 3
    #rbf = 'polynomial'
    periodic_soap = SOAP(
        species=species,
        rcut=rcut,
        nmax=nmax,
        lmax=nmax,
        periodic=True,
        sparse=False,
        average = "inner"
    )
    
    return periodic_soap.create(ASEobject)


