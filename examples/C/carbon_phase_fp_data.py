import sys
sys.path.append('/global/project/projectdirs/m3838/l2g_code/l2g/L2G-LAMMPS/examples/C')

from getSoap import Get_SOAP
from pymatgen.core import Structure

import glob
import natsort
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.io.lammps.data import LammpsData
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise 


class carbon_fp():
    nbins = 50
    other_phases = ['hex', 'diamond', 'graphite','amorphous']
    cols = ['g','m','gray','r']

    def __init__(self):
        self.phases = None
        self.fp_data = None
        self.scaler = None
        self.Xcols = None
        
    def get_distances(self):
        self.Xcols = self.fp_data.columns[self.fp_data.columns.str.contains('soap_fp')]
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(self.fp_data[self.Xcols].values)

        self.Ds_cos = pd.DataFrame(pairwise.cosine_distances(scaled_data, scaled_data),
                                   index=self.fp_data['refined_fname'], columns=self.fp_data['refined_fname'])
        self.Ds_euclidean = pd.DataFrame(pairwise.euclidean_distances(scaled_data, scaled_data),
                                    index=self.fp_data['refined_fname'], columns=self.fp_data['refined_fname'])        
        
        return None
    
    def plot_hist(self, phase):
        fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(9,3.5))

        _ = ax1.hist(self.Ds_euclidean[phase], bins=self.nbins, density=True)
        _ = ax2.hist(self.Ds_cos[phase], bins=self.nbins, density=True)

        for idx,oph in enumerate(self.other_phases):
            ax1.axvline(x = self.Ds_euclidean.loc[phase,oph], label=oph, c=self.cols[idx])
            ax2.axvline(x = self.Ds_cos.loc[phase,oph], label='D %s'%oph, c=self.cols[idx])

        ax1.set_xlabel('Eucleadian dist.', fontsize=12)
        ax2.set_xlabel('Cosine dist.', fontsize=12)

        ax1.set_ylabel('Norm. freq.', fontsize=12)
        ax1.legend(frameon=False, ncol=1)
        
        plt.show()
        
        return None

    def new_phase_distance(self,fname, make_plot=False):
        new_s = LammpsData.from_file(fname,atom_style="atomic").structure
        new_s_fp = Get_SOAP(new_s)

        X_imp_phases = self.scaler.transform(self.fp_data[self.Xcols])
        X_new_phase = self.scaler.transform(new_s_fp)

        Ds_cos = pd.DataFrame(pairwise.cosine_distances(X_new_phase, X_imp_phases), columns=self.fp_data['refined_fname'])
        Ds_euclidean = pd.DataFrame(pairwise.euclidean_distances(X_new_phase, X_imp_phases), columns=self.fp_data['refined_fname'])
        
        
        if make_plot:

            fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=(9,3.5))

            _ = ax1.hist(Ds_euclidean, bins=self.nbins, density=True)
            _ = ax2.hist(Ds_cos, bins=self.nbins, density=True)

            for idx,oph in enumerate(self.other_phases):
                ax1.axvline(x = Ds_euclidean[oph].values, label=oph, c=self.cols[idx])
                ax2.axvline(x = Ds_cos[oph].values, label=oph, c=self.cols[idx])

            ax1.set_xlabel('Eucleadian dist.', fontsize=12)
            ax2.set_xlabel('Cosine dist.', fontsize=12)

            ax1.set_ylabel('Norm. freq.', fontsize=12)
            ax1.legend(frameon=False, ncol=1)

            plt.show()
        
        return Ds_euclidean, Ds_cos

    
if __name__ == "__main__":    
    fnames = glob.glob('MLmodel_POSCARs/*POSCAR*')
    fnames = natsort.natsorted(fnames)

    fps = []
    for fname in fnames:
        s = Structure.from_file(fname)
        fps.append(Get_SOAP(s))


    fnames += ['amorphous']
    s = LammpsData.from_file('/global/project/projectdirs/m3838/l2g_code/l2g/L2G-LAMMPS/examples/C/in.data',atom_style="atomic").structure
    fps.append(Get_SOAP(s))

    fp_data = pd.DataFrame(np.vstack(fps))
    fp_data.columns = ['soap_fp_%s'%c for c in fp_data.columns]
    fp_data['fname'] = fnames
    fp_data['refined_fname'] = fp_data['fname'].apply(lambda x: x.split('/')[-1].split('_')[0])
    
    cfp = carbon_fp()
    cfp.fp_data = fp_data
    cfp.get_distances()
    cfp.plot_hist('diamond')
    
    fname = '/global/project/projectdirs/m3838/l2g_code/l2g/L2G-LAMMPS/examples/C/gen_10/output/out-1.data'
    Ds_euclidean, Ds_cos = cfp.new_phase_distance(fname, make_plot=1)
    
    with open('C_imp_phase_fp.pkl', 'wb') as f:
        pickle.dump(cfp,f)