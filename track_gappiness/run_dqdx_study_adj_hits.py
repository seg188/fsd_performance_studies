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
import time
_version_tag = 'none'

geo = None

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


def main(in_fname, out_fname, version_tag, use_adjacent_hits):
   
    start=time.time()
    global _version_tag
    _version_tag = version_tag

    global geo
    try:
        geo = load_geo('geo-{}.json'.format(version_tag))
    except:
        raise RuntimeError('Invalid version tag! {}'.format(version_tag))
    # create output file
    create_h5_file_safe(out_fname, ['y', 'z', 'trim_mean', 'trim_std', 'module', 'anode', 'mu_fit', 'eta_fit', 'fit_flag'])

    print('==> Created output file:\t{:0.4f}s'.format(time.time()-start))
    # Load data file 
    f_nom = h5py.File(in_fname)
    print('==> Opened input file:\t{:0.4f}s'.format(time.time()-start))
    
    NHITS=f_nom['pixel_length'].shape[0]-1

    lengths = np.array(f_nom['pixel_length'][0:NHITS])
    
    print('==> Loaded Dataset pixel_length:\t{:0.4f}s'.format(time.time()-start))

    anodes = np.array(f_nom['anode'][0:NHITS])
    

    print('==> Loaded Dataset anode:\t{:0.4f}s'.format(time.time()-start))

    modules = np.array(f_nom['module'][0:NHITS])
    
    print('==> Loaded Dataset module:\t{:0.4f}s'.format(time.time()-start))
    
    angles = np.array(f_nom['angle'][0:NHITS])
        
    print('==> Loaded Dataset angle:\t{:0.4f}s'.format(time.time()-start))

    
    dQs = np.array(f_nom['total_charge_collected'][0:NHITS])
    
    print('==> Loaded Dataset total_charge_collected:\t{:0.4f}s'.format(time.time()-start))

    py, pz = None, None
 
    if use_adjacent_hits:
        print('using adjacent hits!')
        py = np.array(f_nom['pix_approach_pt_y'][0:NHITS])
        print('==> Loaded Dataset pix_approach_pt_y:\t{:0.4f}s'.format(time.time()-start))
        pz = np.array(f_nom['pix_approach_pt_z'][0:NHITS])
        print('==> Loaded Dataset pix_approach_pt_z:\t{:0.4f}s'.format(time.time()-start))
    else:

        py = np.array(f_nom['pixel_y'][0:NHITS])
        print('==> Loaded Dataset pixel_y:\t{:0.4f}s'.format(time.time()-start))
        pz = np.array(f_nom['pixel_z'][0:NHITS])
        print('==> Loaded Dataset pixel_z:\t{:0.4f}s'.format(time.time()-start))
    
    # get unique event ids for each set of hits by breaking at changes in angle, total_length

    event_breaks =  (np.absolute(np.diff( angles )) > 1e-4).astype(int)
    event_ids = np.concatenate( ( np.array([0]), np.cumsum(event_breaks)) ) 
    
    print('==> Assigned Event IDs:\t{:0.4f}s'.format(time.time()-start))
    
    if _version_tag=='2x2': 
        loop_group = range(1,9)
    elif _version_tag=='fsd':
        loop_group=range(1,3)
    else:
        raise RuntimeError('Detector version (2x2, fsd) not specified correctly!')
    
    for io_group in tqdm(loop_group):
        
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

        module=0
        anode=0

        if _version_tag=='2x2': 
        
            module = (io_group-1)//2
            anode = (io_group+1) % 2 + 1

        elif _version_tag=='fsd':
            module = 0
            anode = io_group

        
        
        mp = (anodes  == anode)&\
             (modules== module)
    
        print('==> Created intial module/anode mask:\t{:0.4f}s'.format(time.time()-start))
        
        _lengths = lengths[mp]
        _angles = angles[mp]
        _dQs = dQs[mp]
        _py = py[mp]
        _pz = pz[mp]
        _event_ids = event_ids[mp]
    
        print('==> Sliced data for module/anode:\t{:0.4f}s'.format(time.time()-start))
        
        mp = (_lengths > pixel_pitch*0.25) & (_lengths / np.sin(_angles) > pixel_pitch)& (_lengths / np.sin(_angles) < 3*pixel_pitch)
        if use_adjacent_hits:
            mp = mp | (_lengths < 0)
    
        mp = mp & ( ~( (_py==-1) & (_pz==-1) )  )

        print('==> Created mask for relevant hits:\t{:0.4f}s'.format(time.time()-start))
        _lengths = _lengths[mp]
        _angles = _angles[mp]
        _dQs = _dQs[mp]
        _py = _py[mp]
        _pz = _pz[mp]
        _event_ids = _event_ids[mp]

        print('==> Sliced hits from mask, ready to start fitting:\t{:0.4f}s'.format(time.time()-start))
        print_count=0
        
        for ybin in tqdm(range(10, len(pix_y)-10), leave=False):
            my = (np.absolute(_py - pix_y[ybin]) < pixel_pitch/3)
    
            _y_pz = _pz[my]
            _y_dQs = _dQs[my]
            _y_lengths = (_lengths / np.sin(_angles))[my] 
            _y_event_ids = _event_ids[my]
            
            for zbin in range(10,len(pix_z)-10):

                mz  = (np.absolute(_y_pz - pix_z[zbin]) < pixel_pitch/3)

                if np.sum(mz) < 3: 
                    print('fewer than 3 hits! continuing')
                    continue

                _z_lengths = _y_lengths[mz]

                _z_Q = _y_dQs[mz]
                _z_event_ids = _y_event_ids[mz]

                evs, counts = np.unique(_z_event_ids, return_counts=True)
                
                dQ=[]
                dx=[]

                if any(counts > 1):
                    for ev in evs:
                        mm = _z_event_ids==ev

                        l = np.max(_z_lengths[mm])

                        if l < 0:
                            #print('no positive lengths!')
                            #print(_z_lengths[mm])
                            continue

                        dQ.append( np.sum(_z_Q[mm]) )
                        dx.append(l)

                    dQ=np.array(dQ)
                    dx=np.array(dx)

                else:

                    mm = _z_lengths > 0
                    dQ = _z_Q[ mm ]
                    dx = _z_lengths[ mm ]

                
                dQdx = dQ/dx

                params=None
                
                if np.sum(mz) > 40:
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
        parser.add_argument('--use_adjacent_hits', action='store_true', default=False, help='''Use hits adjacent to main track fit''')
        parser.add_argument('--version_tag', '-t', default='2x2', type=str, help='''(2x2 or fsd) which detector is being analyzed''')
        args = parser.parse_args()
        main(**vars(args))

