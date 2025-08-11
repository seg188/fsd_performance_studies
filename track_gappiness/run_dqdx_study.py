import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import dereference
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy
import warnings
from geometry import load_geo
import argparse
import pylandau
from scipy.optimize import curve_fit


geo = load_geo()

warnings.filterwarnings("ignore")

def trim_std(values, pct):
    sorted_values = np.sort(values)
    num_trunc = int(pct*values.shape[0])
    return np.std( sorted_values[num_trunc:-num_trunc]  )

def landau(x, a, b, c):
    # x, mpv, eta, sigma, A
    return pylandau.landau(x, a, b, c)

def create_h5_file_safe(file_name, data_format):
    '''Create an HDF5 file, don't overwrite if the file already exists'''
    if os.path.isfile(file_name):
        print('Not creating file, already exists!')
        return

    with h5py.File(file_name, "w") as f:
        for key in data_format:
            f.create_dataset(key, data=np.array([]), compression="gzip", maxshape=(None,))

    return


def main(in_fname, out_fname):

    # create output file
    create_h5_file_safe(out_fname, ['y', 'z', 'trim_mean', 'trim_std', 'module', 'anode', 'mu_fit', 'eta_fit', 'fit_flag'])

    # Load data file 
    f_nom = h5py.File(in_fname)

    lengths = np.array(f_nom['pixel_length'][:])
    anodes = np.array(f_nom['anode'][:])
    modules = np.array(f_nom['module'][:])
    angles = np.array(f_nom['angle'][:])
    dQs = np.array(f_nom['total_charge_collected'][:])
    py = np.array(f_nom['pixel_y'][:])
    pz = np.array(f_nom['pixel_z'][:])

    for io_group in tqdm(range(1,9)):
        
        # pixel y position
        y_dqdx=[]

        #pixel z position
        z_dqdx=[]

        #landau mu from fit (if fit is performed, else -1)
        mu_dqdx=[]

        #landau eta from fit (if fit is performed, else -1)
        eta_dqdx=[]

        # truncated mean of dq/dx
        trim_mean_dqdx=[]

        #truncated std of dqdx
        trim_std_dqdx=[]

        #module of channel
        module_dqdx = []
    
        #anode of channel
        anode_dqdx = []

        #flag for if fit was performed and converged
        # 1 - converged, 0 - not converged, -1 - not attempted
        fit_flag_dqdx = []
    

        
        pixel_pitch = geo[io_group]['pixel_pitch']
        pix_y, pix_z = geo[io_group]['pix_y_z']

        module = (io_group-1)//2
        anode = (io_group+1) % 2 + 1

        mp = (anodes  == anode)&\
             (modules== module)
    
        _lengths = lengths[mp]
        _angles = angles[mp]
        _dQs = dQs[mp]
        _py = py[mp]
        _pz = pz[mp]
    
        mp = (_lengths > pixel_pitch*0.7)& (_lengths / np.sin(_angles) > pixel_pitch)& (_lengths / np.sin(_angles) < 3*pixel_pitch)
    
        _lengths = _lengths[mp]
        _angles = _angles[mp]
        _dQs = _dQs[mp]
        _py = _py[mp]
        _pz = _pz[mp]
        
        for ybin in tqdm(range(10, len(pix_y)-10), leave=False):
            my = (np.absolute(_py - pix_y[ybin]) < pixel_pitch/2)
    
            _y_pz = _pz[my]
            _y_dQs = _dQs[my]
            _y_full_lengths = ((_lengths / np.sin(_angles)))[my] 
            
            for zbin in range(10,len(pix_z)-10):
    
                mz  = (np.absolute(_y_pz - pix_z[zbin]) < pixel_pitch/2)
    
                dQdx = (_y_dQs / _y_full_lengths)[mz]
    
                params=None
                
                if np.sum(mz) > 40 and False:
                    try:
                        vals, bins = np.histogram(dQdx, bins=25, range=(0,100))
                        params, pcov = curve_fit(landau, (bins[1:]+bins[:-1])/2, vals, p0=[40, 10, 5], bounds=(0,100) )
                        fit_flag_dqdx.append(1)
                    except:
                        params=None
                        fit_flag_dqdx.append(0)
    
                else:
                    fit_flag_dqdx.append(-1)
                    
    
                if params is None:
                    mu_dqdx.append( np.nan )
                    eta_dqdx.append( np.nan  )
                else:
                    mu_dqdx.append( params[0] )
                    eta_dqdx.append( params[1]  )
    
                y_dqdx.append(pix_y[ybin])
                z_dqdx.append(pix_z[zbin])
                trim_mean_dqdx.append( scipy.stats.trim_mean( dQdx,  0.3) )
                trim_std_dqdx.append( trim_std( dQdx,  0.3) )
                module_dqdx.append(module)
                anode_dqdx.append(anode)

        #save everything to the data file!
        dstore={
            'y' : y_dqdx,
            'z' : z_dqdx,
            'trim_mean' : trim_mean_dqdx,
            'trim_std' : trim_std_dqdx,
            'anode' : anode_dqdx,
            'module': module_dqdx,
            'mu_fit' : mu_dqdx,
            'eta_fit' : eta_dqdx,
            'fit_flag' : fit_flag_dqdx
            }
        
        with h5py.File(out_fname, 'a') as hf:
            for key in dstore.keys():
                d=np.array(dstore[key])
                orig_size = hf[key].shape[0]
                hf[key].resize(( orig_size + d.shape[0]), axis = 0)
                hf[key][-d.shape[0]:] = d

    print('Completed! Data saved to {}'.format(out_fname))
    return


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--in_fname', '-i', default='./', type=str, help='''Directory containing output file from efficiency study (run_track_efficiency_study.py)''')
        parser.add_argument('--out_fname', '-o', default='default_outf_dqdx.h5', type=str, help='''Name/path for output to be written''')
        args = parser.parse_args()
        main(**vars(args))

