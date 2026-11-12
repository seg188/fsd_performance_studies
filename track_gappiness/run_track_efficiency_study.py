import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import dereference
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy
import argparse
import json
import warnings
from copy import deepcopy
from data_format import data_store_nominal
import fit_tracks
from fit_tracks import fit_line_3d

from geometry import load_geo


warnings.filterwarnings("ignore")


global_event_id = -1
_version_tag='none'
geo = {}

###############################################################################################

###
# mapping (io_group, io_channel, chip_id, channel_id) <--> unique identifier
def unique_to_channel_id(unique):
    return (unique % 100)

def unique_to_chip_id(unique):
    return ((unique// 100) % 1000)

def unique_to_io_channel(unique):
    return ((unique//(100*1000)) % 1000)

def unique_to_io_group(unique):
    return ((unique // (100*1000*1000)) % 1000)

def unique_chip_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000 \
            + d['chip_id'].astype(int))*100

def unique_channel_id(d):
    return ((d['io_group'].astype(int)*1000+d['io_channel'].astype(int))*1000 \
            + d['chip_id'].astype(int))*100 + d['channel_id'].astype(int)
###

###
# Data format definition and some useful hdf5 wrapper functions

import gc
def close_all_h5_files():
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed

def create_h5_file_safe(file_name, data_format=data_store_nominal.keys()):
    '''Create an HDF5 file, don't overwrite if the file already exists'''
    if os.path.isfile(file_name):
        print('Not creating file, already exists!')
        return

    with h5py.File(file_name, "w") as f:
        for key in data_format:
            f.create_dataset(key, data=np.array([]), compression="gzip", maxshape=(None,))

    return

###

###
# Some useful geometry helper functions
# Note pix_ij, 'pixel_id' is NOT equivalent to unique_channel_id above

def get_pixel_yz(i,j, io_group):
    '''Return (y,z) of pixel at index (i,j) in specified io_group'''
    pix_y, pix_z = geo[io_group]['pix_y_z']
    return pix_y[i], pix_z[j]

def get_closest_pix_ij(y, z, io_group):
    '''Return i,j index of closest pixel to position (y,z)'''
    pix_y, pix_z = geo[io_group]['pix_y_z']
    closest_y = np.argmin( np.absolute(pix_y - y) )
    closest_z = np.argmin( np.absolute(pix_z - z) )
    return closest_y, closest_z

def get_unique_pixel_id(i, j):
    '''Return unique single integer encoding i,j indices, only unique within io_group'''
    return i*100000 + j

def get_unique_pixel_ids(ii):
    '''Array version of get_unique_pixel_id'''
    return ii[:,0]*1000000 + ii[:,1]

def get_unique_pixel_ids_yz( yz, io_group ):
    '''Return unique single integer encoding i,j indices from yz positions. yz is an array here.'''
    min_y, min_z = geo[io_group]['min_y_z']
    pixel_pitch = geo[io_group]['pixel_pitch']
    i = np.rint((yz[:,0]-min_y)/pixel_pitch).astype(int)
    j = np.rint((yz[:,1]-min_z)/pixel_pitch).astype(int)
    return i*1000000 + j
    
def get_ij_from_id(id_val):
    '''inverse for id-->(i,j)'''
    return id_val//1000000 , id_val % 1000000

def get_closest_pixel_yz(y, z, io_group):
    '''Get pixel center position closest to point (y,z)'''
    i, j = get_closest_pix_ij(y, z, io_group)
    return get_pixel_yz(i, j, io_group)

def get_all_hit_pixels(start, finish, io_group):
    '''Get an array of all pixel (y,z) positions crossed by a track passing from pt. start-->finish.

        Returns <pixel yz position array>, <length of track crossing pixel (projected in 2D)>

        Crossing length is done by histogramming discrete points so crossing lengths < pixel_pitch/10 might be missed.

        All hits must be in the same io_group.
    '''
    direction = finish - start    
    length = np.linalg.norm(direction)
    direction = direction/length

    pixel_pitch = geo[io_group]['pixel_pitch']
    

    n_pts_to_test = int( (length / pixel_pitch) * 10)+1
    line_pts = np.linspace(0, length, n_pts_to_test)
    
    # get pixels which are hit
    hit_y_z = np.zeros( ( n_pts_to_test, 2 )  )
    for i in range(n_pts_to_test):
        y, z = start[0] + line_pts[i]*direction[0], start[1] + line_pts[i]*direction[1] 
        _y, _z = get_closest_pixel_yz(y, z, io_group)
        hit_y_z[i,0] =  _y
        hit_y_z[i,1] =  _z

    pyz, counts = np.unique(hit_y_z, axis=0, return_counts=True)
    mask = (pyz[:,0] > np.min(pix_y)) & (pyz[:,0] < np.max(pix_y)) & (pyz[:,1] > np.min(pix_z)) & (pyz[:,1] < np.max(pix_z))

    pyz=pyz[mask]
    counts = counts[mask]

    return pyz, length*(counts / np.sum(counts))


def get_all_hit_pixels_kinked(starts, finishs, io_group, trans_diff=None):
    '''Get an array of all pixel (y,z) positions crossed by a track passing from pt. start-->finish.

        For this _kinked version, starts, finishs are lists of pts defining the segments of track

        Returns <pixel yz position array>, <length of track crossing pixel (projected in 2D)>

        Crossing length is done by histogramming discrete points so crossing lengths < pixel_pitch/10 might be missed.
    '''
    pixel_pitch = geo[io_group]['pixel_pitch']


    pix_y, pix_z = geo[io_group]['pix_y_z']
    hit_y_z = np.array([])
    full_length=0

    def get_trans_diff():
        if trans_diff is None: return 0.0
        else:
            return np.random.normal(loc=0.0, scale=trans_diff)
    
    for i in range(len(starts)):
        start = starts[i][1:]
        finish = finishs[i][1:]
    
        direction = finish - start    
        length = np.linalg.norm(direction)
        
        direction = direction/length
    
        n_pts_to_test_float = ( (length / pixel_pitch) * 12)

        if np.isnan(n_pts_to_test_float): 
            return np.array([]), np.array([])

        n_pts_to_test=int(n_pts_to_test_float)
        line_pts = np.linspace(0, length, n_pts_to_test)
    
        # get pixels which are hit
        _hit_y_z = np.zeros( ( n_pts_to_test, 2 )  )
        for i in range(n_pts_to_test):
            y, z = start[0] + line_pts[i]*direction[0] + get_trans_diff(), start[1] + line_pts[i]*direction[1] + get_trans_diff()
            _y, _z = get_closest_pixel_yz(y, z, io_group)
        
            _hit_y_z[i,0] =  _y
            _hit_y_z[i,1] =  _z

        if hit_y_z.shape[0]==0:
            hit_y_z = _hit_y_z
        else:
            hit_y_z = np.concatenate( (hit_y_z, _hit_y_z) )

        full_length+=length

    pyz, counts = np.unique(hit_y_z, axis=0, return_counts=True)
    mask = (pyz[:,0] > np.min(pix_y)) & (pyz[:,0] < np.max(pix_y)) & (pyz[:,1] > np.min(pix_z)) & (pyz[:,1] < np.max(pix_z))

    pyz=pyz[mask]
    counts = counts[mask]

    return pyz, full_length*(counts / np.sum(counts))

def get_closest_approach_to_track_kinked(y, z, starts, finishs):
    '''
    Get approach length and point of closest approach of the point (y, z) to the kinked track encoded by starts, finishes.
    '''

    # get closest apprach to each pair of start, end, and take minimum

    lengths = []
    closest_approaches = []
            
    for i in range(len(starts)):
        start = starts[i][1:]
        finish = finishs[i][1:]
    
        direction = finish - start    
        length = np.linalg.norm(direction)
        
        direction = direction/length

        _y, _z = y-start[0], z-start[1]
        
        pt_length = np.sqrt( _y**2 + _z**2 )
        
        dot = (_y*direction[0] + _z*direction[1]) / pt_length

        theta = np.arccos(dot)

        length_to_track = np.sin(theta) * pt_length
        
        pt_of_closest_approach = (start[0] + direction[0] * np.cos(theta) * pt_length, start[1] + direction[1] * np.cos(theta) * pt_length  )

        lengths.append( length_to_track  )
        closest_approaches.append( pt_of_closest_approach )

    l_arg = np.argmin(lengths)
    
    return lengths[l_arg], closest_approaches[l_arg]


def get_closest_approach_to_track_kinked_3D(x, y, z, starts, finishs):
    '''
    Get approach length and point of closest approach of the point (y, z) to the kinked track encoded by starts, finishes.
    '''

    # get closest apprach to each pair of start, end, and take minimum

    lengths = []
    closest_approaches = []
            
    for i in range(len(starts)):
        start = starts[i]
        finish = finishs[i]
    
        direction = finish - start    
        length = np.linalg.norm(direction)
        
        direction = direction/length

        _x, _y, _z = x-start[0], y-start[1], z-start[2]
        
        pt_length = np.sqrt( _x**2 + _y**2 + _z**2 )
        
        dot = (_x*direction[0] + _y*direction[1] + _z*direction[2]) / pt_length

        theta = np.arccos(dot)

        length_to_track = np.sin(theta) * pt_length
        
        pt_of_closest_approach = (start[0] + direction[0] * np.cos(theta) * pt_length, start[1] + direction[1] * np.cos(theta) * pt_length, start[2] + direction[2] * np.cos(theta) * pt_length  )

        lengths.append( length_to_track  )
        closest_approaches.append( pt_of_closest_approach )

    l_arg = np.argmin(lengths)
    
    return lengths[l_arg], closest_approaches[l_arg]


def find_tracks(file, min_event_nhit=40, use_prompt=False, exclude_disabled=True):

    global _version_tag, global_event_id

    # initialize data dictionary
    data_store = {}
    for key in data_store_nominal.keys(): data_store[key] = []

    # open file data/references
    f = h5py.File(file)

    hit_type_name=''
    if _version_tag=='2x2':
        hit_type_name='calib_prompt_hits'
    elif _version_tag=='fsd':
        hit_type_name='calib_prompt_hits' 

    events = f['charge/events/data']['id'][ f['charge/events/data']['nhit']>min_event_nhit]
    events_hits_ref = f['charge/events/ref/charge/{}/ref'.format(hit_type_name)][:]
    events_hits_rref = f['charge/events/ref/charge/{}/ref_region'.format(hit_type_name)][:]
    hit_data = f['charge/{}/data'.format(hit_type_name)][:]

    events_trig_rref = np.array(f['charge/raw_events/ref/charge/ext_trigs/ref'][:])
    ext_trig_data = f['charge/ext_trigs/data'][:]

    if use_prompt and not (_version_tag=='fsd') :
        events_phits_ref = f['charge/events/ref/charge/calib_prompt_hits/ref'][:]
        events_phits_rref = f['charge/events/ref/charge/calib_prompt_hits/ref_region'][:]
        phit_data = f['charge/calib_prompt_hits/data'][:]
       
    #loop over all relevant events    
    for ev in tqdm( events, leave=False, desc='parsing events...'):
        #add tag to data file for beam / cosmic event

        trigs = dereference.dereference(ev, events_trig_rref, ext_trig_data).flatten()
        trig_tag=-1
        if 5 in trigs['iogroup'].astype(int):
            trig_tag=5
        elif 6 in trigs['iogroup'].astype(int):
            trig_tag=6
        
        ev_hits = dereference.dereference(ev, events_hits_ref, hit_data, region=events_hits_rref).flatten()

        if exclude_disabled: ev_hits = ev_hits[ ~(ev_hits['is_disabled'][:])  ]
       
        if _version_tag=='2x2': 
            #anode for 2x2
            loop_group = range(1,9)
        elif _version_tag=='fsd':
            # anode for fsd
            loop_group=range(1,3)
        else:
            raise RuntimeError('Detector version (2x2, fsd) not specified correctly!')
        
        for io_group in loop_group:
            global_event_id += 1
        
            pixel_pitch = geo[io_group]['pixel_pitch']
            module = (io_group-1)//2
            anode = (io_group+1) % 2 + 1
            
            hits = ev_hits.copy()
            
            if _version_tag=='2x2':
                mog = hits['io_group'].astype(int) == io_group
            elif _version_tag=='fsd':
                mog = ((hits['io_group'].astype(int)+1)//2) == io_group
            

            #require at least 20 hits total in this io group
            #if all hits in a line, ~8cm of track
            if np.sum(mog)<20: continue
                
            _X = np.array([hits['x'][mog], hits['y'][mog], hits['z'][mog]]).transpose()
            _Q = np.array(hits['Q'][mog])
            all_hit_io_group = np.array(hits['io_group'][:][mog])
            all_hit_io_chan = np.array(hits['io_channel'][:][mog])
            all_hit_chip_id = np.array(hits['chip_id'][:][mog])
            all_hit_chan_id = np.array(hits['channel_id'][:][mog])
            #_backtrack = backtrack[mog]

            # Identify rows with NaN values
            rows_with_nan = np.isnan(_X).any(axis=1)

            # Invert the boolean array to select rows without NaN values
            rows_without_nan = ~rows_with_nan

            # error in geometry parsing... fixed all known bugs which result in this, but keeping it here to stop crashes
            if np.sum(~rows_without_nan) > 0:
                continue
                
            # Filter the array to remove rows with NaN
            #_X = _X[rows_without_nan]
            #_Q = _Q[rows_without_nan]
            #all_hit_io_group = all_hit_io_group[rows_without_nan]
            #all_hit_io_chan = all_hit_io_chan[rows_without_nan]
            #all_hit_chip_id = all_hit_chip_id[rows_without_nan]
            #all_hit_chan_id = all_hit_chan_id[rows_without_nan]
            
            if _X.shape[0]<20: continue
            pca=PCA(n_components=3)
            pca.fit(_X)

            #get object is roughly track-like
            if np.max(pca.explained_variance_ratio_) < 0.965: continue

            #fit all hits to a line
            fcentr, fdirection = fit_line_3d(_X)

            #try to remove delta rays or non-contiguous hits
            #make sure all sub-sections of the track also fit along this line
            subdir, subcent = [], []
            costhet=[]
            clust_label=[]
            skip=False
            clustering = DBSCAN(eps=20*pixel_pitch, min_samples=2).fit(_X)
            
            for clust in set(clustering.labels_):
                if clust < 0: continue
                m=np.array(clustering.labels_==clust)

                #ensure at least 12 hits in the cluster
                if np.sum(m)<12:continue
                X = _X[m]
                Q = _Q[m]

                hit_io_group=all_hit_io_group[m]
                hit_io_chan=all_hit_io_chan[m]
                hit_chip_id=all_hit_chip_id[m]
                hit_chan_id=all_hit_chan_id[m]


                index_results, used_mask = fit_tracks.fit_tracks(X)

                if index_results is None: continue

                if len(index_results.keys())==0: continue

                filt_mask = used_mask > -999

                X_filt = X[filt_mask]
                Q_filt = Q[filt_mask]
                hit_io_group_filt = hit_io_group[filt_mask]
                hit_io_chan_filt =  hit_io_chan[filt_mask]
                hit_chip_id_filt = hit_chip_id[filt_mask]
                hit_chan_id_filt = hit_chan_id[filt_mask]

                #SWAP IN PROMPT HITS if use_prompt!
                ##################################
                if use_prompt and not (_version_tag=='fsd'):
                    hits = dereference.dereference(ev, events_phits_ref, phit_data, region=events_phits_rref).flatten()
                    if exclude_disabled: hits = hits[ ~(hits['is_disabled'][:])  ]
                    
                    if _version_tag=='2x2':
                        mog = hits['io_group'].astype(int) == io_group
                    elif _version_tag=='fsd':
                        mog = ((hits['io_group'].astype(int)+1)//2) == io_group

                    X = np.array([hits['x'][mog], hits['y'][mog], hits['z'][mog]]).transpose()
                    Q = np.array(hits['Q'][mog])
                    hit_io_group = np.array(hits['io_group'][:][mog])
                    hit_io_chan = np.array(hits['io_channel'][:][mog])
                    hit_chip_id = np.array(hits['chip_id'][:][mog])
                    hit_chan_id = np.array(hits['channel_id'][:][mog])

                    # now filter out hits that are too far from this track (>1 pixel pitch)
                    dist_to_track = np.ones( Q.shape  )*9999999.

                    all_starts, all_ends= [], []
                    for index in index_results.keys():
                        start, end = index_results[index]['start'], index_results[index]['end']
                        all_starts.append(start)
                        all_ends.append(end)

                    for i in range(Q.shape[0]):
                        x, y, z = X[i, 0], X[i, 1], X[i, 2]
                        ll, pp = get_closest_approach_to_track_kinked_3D(x, y, z, all_starts, all_ends)
                        dist_to_track[i] = ll
                    
                    filt_mask = dist_to_track < 2*pixel_pitch

                    X_filt = X[filt_mask]
                    Q_filt = Q[filt_mask]
                    hit_io_group_filt = hit_io_group[filt_mask]
                    hit_io_chan_filt =  hit_io_chan[filt_mask]
                    hit_chip_id_filt = hit_chip_id[filt_mask]
                    hit_chan_id_filt = hit_chan_id[filt_mask]
                    
                ##################################

                
                pyz, l = None, np.array([])
                angles = []

                for index in index_results.keys():
                    start, end = index_results[index]['start'], index_results[index]['end']
                    ###
                    _pyz, _l = get_all_hit_pixels_kinked( [start], [end], io_group)

                    ##

                    if pyz is None:
                        pyz = _pyz
                    else:
                        pyz = np.concatenate( ( pyz, _pyz )  )
                    l = np.concatenate( (l, _l)  )

                    angles += [index_results[index]['angle']]*_pyz.shape[0]
                    
                if pyz is None: continue
                if pyz.shape[0]<1: continue    

                # find which pixels were MISSED
                all_hit_pixel_ids = get_unique_pixel_ids_yz( X_filt[:,1:], io_group )
                hit_pixel_ids, hit_idx, hit_counts = np.unique(all_hit_pixel_ids, return_index=True, return_counts=True )
            
                predicted_pixel_ids = get_unique_pixel_ids_yz(  pyz, io_group  )
                #Now, save summary information, as well as pixel-by-pixel information

                all_predicted_and_hit = set( list(predicted_pixel_ids) + list(all_hit_pixel_ids))
                
                N=len(all_predicted_and_hit)

                _total_charge_collected = [-1]*N
                _hit_drift = [-1]*N
                _is_hit = [False]*N
                _hit_io_group = [-1]*N
                _hit_io_chan = [-1]*N
                _hit_chip_id = [-1]*N
                _hit_chan_id = [-1]*N
                _approach_length = [-1]*N
                _approach_pt_y = [-1]*N
                _approach_pt_z = [-1]*N
                _pix_approach_pt_y = [-1]*N
                _pix_approach_pt_z = [-1]*N
                _n_hits = [-1]*N
                _true_q = [-1]*N
                _true_length = [-1]*N
                
                _pixel_id = [-1]*N
                _pixel_y = [np.nan]*N
                _pixel_z = [np.nan]*N
                _pixel_length = [np.nan]*N
                
                for ipid, pid in enumerate(list( all_predicted_and_hit )):

                    ind = -1
                    if pid in predicted_pixel_ids:
                        ind = list(predicted_pixel_ids).index(pid)
                        _y = pyz[ind, 0]
                        _z = pyz[ind, 1]
                        pl = l[ind]
                    else:
                        ind = list(all_hit_pixel_ids).index(pid)
                        _y = X_filt[ind, 1]
                        _z = X_filt[ind, 2]
                        pl = 0

                    _pixel_id[ipid] = pid      
                    _pixel_length[ipid] = pl
                    _pixel_y[ipid] = _y
                    _pixel_z[ipid] = _z
                    
                    if pid in hit_pixel_ids:
                        hit_mask = all_hit_pixel_ids==pid
                        fit_index = used_mask[hit_mask][0]
                            
                        indxs = np.where(hit_mask)[0]
                        _total_charge_collected[ipid] =  np.sum(Q_filt[indxs]) 
                        _n_hits[ipid] = indxs.shape[0]
                        _hit_drift[ipid] = np.mean(X_filt[:,0][indxs]) 
                        _is_hit[ipid] = True
                        _hit_io_group[ipid] = hit_io_group_filt[indxs[0]] 
                        _hit_io_chan[ipid] = hit_io_chan_filt[indxs[0]]
                        _hit_chip_id[ipid] = hit_chip_id_filt[indxs[0]]
                        _hit_chan_id[ipid] = hit_chan_id_filt[indxs[0]]

                        if not fit_index in index_results.keys():
                            continue
                            
                        approach_length, approach_pt = get_closest_approach_to_track_kinked( X_filt[:,1][indxs][0], X_filt[:,2][indxs][0],\
                                                         [ index_results[fit_index]['start'] ], [ index_results[fit_index]['end'] ] )

                        _approach_length[ipid] = approach_length
                        _approach_pt_y[ipid] = approach_pt[0]
                        _approach_pt_z[ipid] = approach_pt[1]

                        pix_y_approach_pt, pix_z_approach_pt = get_closest_pixel_yz(approach_pt[0], approach_pt[1], io_group)

                        _pix_approach_pt_y[ipid] = pix_y_approach_pt
                        _pix_approach_pt_z[ipid] = pix_z_approach_pt
                        

                data_store['total_charge_collected'] += _total_charge_collected
                data_store['hit_drift'] += _hit_drift
                data_store['is_hit'] += _is_hit
                data_store['hit_io_group'] += _hit_io_group
                data_store['hit_io_chan'] += _hit_io_chan
                data_store['hit_chip_id'] += _hit_chip_id
                data_store['hit_chan_id'] += _hit_chan_id
                data_store['approach_length'] += _approach_length
                data_store['approach_pt_y'] += _approach_pt_y
                data_store['approach_pt_z'] += _approach_pt_z
                data_store['pix_approach_pt_y'] += _pix_approach_pt_y
                data_store['pix_approach_pt_z'] += _pix_approach_pt_z
                data_store['true_q'] += _true_q
                data_store['n_hits'] += _n_hits
                data_store['true_length'] += _true_length
                data_store['module']+=[module]*N
                data_store['anode'] += [anode]*N
                data_store['angle'] += [np.mean(angles)]*N
                data_store['trigger_tag'] += [trig_tag]*N
                data_store['unique_track_id'] += [global_event_id]*N

                data_store['pixel_id'] += _pixel_id
                data_store['pixel_y'] += _pixel_y
                data_store['pixel_z'] += _pixel_z
                data_store['pixel_length'] += _pixel_length
                 
                if not len(data_store['pixel_id'])==len(data_store['total_charge_collected']):
                    print('wot wot wot???')                                           
                    
    return data_store

def main(data_dir, outf_name, use_prompt, max_files, version_tag, specify_run):
    
    global _version_tag
    _version_tag = version_tag

    create_h5_file_safe(outf_name)

    data_files = sorted(os.listdir(data_dir))
    data_files = [f for f in data_files if not 'json' in f]
    close_all_h5_files()

    files_to_parse = len(data_files)
    
    ## load geometry
    global geo
    
    if _version_tag=='2x2':
        geo = load_geo('geo-2x2.json')
    elif _version_tag=='fsd':
        geo = load_geo('geo-fsd.json')
    else:
        raise RuntimeError('Detector version (2x2, fsd) not specified correctly!')

    files_to_loop = data_files
    if not specify_run is None:
        files_to_loop = [file for file in data_files if str(specify_run) in file]
    
    if max_files  > 0:
        files_to_loop = files_to_loop[max_files:]

    if True:
        for file in tqdm(files_to_loop, desc='parsing files...'):
            
            dstore = find_tracks(data_dir + file, use_prompt=use_prompt) 

            # save data from dstore
            with h5py.File(outf_name, 'a') as hf:
                for key in dstore.keys():
                    d=np.array(dstore[key])
                    orig_size = hf[key].shape[0]
                    hf[key].resize(( orig_size + d.shape[0]), axis = 0)
                    hf[key][-d.shape[0]:] = d


    print('Completed processing! Data saved to {}'.format(outf_name))

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', '-i', default='./', type=str, help='''Directory containing ndlar_flow files to process''')
        parser.add_argument('--use_prompt', action='store_true', help='''Use prompt hits to study efficiency''')
        parser.add_argument('--outf_name', '-o', default='default_outf.h5', type=str, help='''Name/path for output to be written''')
        parser.add_argument('--specify_run', '-r', default=None, type=str, help='''Run number to process, ignore others. Must be in file name''')
        parser.add_argument('--version_tag', '-t', default='2x2', type=str, help='''Either 2x2 or fsd, determines which geometry and anode/module mapping to use''')
        parser.add_argument('--max_files', default=-1, type=int, help='''Max number of files to process''')
        args = parser.parse_args()
        main(**vars(args))
