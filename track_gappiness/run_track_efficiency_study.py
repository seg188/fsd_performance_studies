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

from geometry import load_geo

geo = load_geo()

warnings.filterwarnings("ignore")

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


def get_all_hit_pixels_kinked(starts, finishs, io_group):
    '''Get an array of all pixel (y,z) positions crossed by a track passing from pt. start-->finish.

        For this _kinked version, starts, finishs are lists of pts defining the segments of track

        Returns <pixel yz position array>, <length of track crossing pixel (projected in 2D)>

        Crossing length is done by histogramming discrete points so crossing lengths < pixel_pitch/10 might be missed.
    '''
    pixel_pitch = geo[io_group]['pixel_pitch']


    pix_y, pix_z = geo[io_group]['pix_y_z']
    hit_y_z = np.array([])
    full_length=0
    
    for i in range(len(starts)):
        start = starts[i][1:]
        finish = finishs[i][1:]
    
        direction = finish - start    
        length = np.linalg.norm(direction)
        
        direction = direction/length
    
        n_pts_to_test_float = ( (length / pixel_pitch) * 10)

        if np.isnan(n_pts_to_test_float): 
            return np.array([]), np.array([])

        n_pts_to_test=int(n_pts_to_test_float)
        line_pts = np.linspace(0, length, n_pts_to_test)
    
        # get pixels which are hit
        _hit_y_z = np.zeros( ( n_pts_to_test, 2 )  )
        for i in range(n_pts_to_test):
            y, z = start[0] + line_pts[i]*direction[0], start[1] + line_pts[i]*direction[1] 
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

def fit_line_3d(points):
    """
    Fits a straight line to an array of 3D points using PCA.
    
    Parameters:
        points (np.ndarray): An Nx3 array of 3D points.

    Returns:
        point_on_line (np.ndarray): A point on the best-fit line (the centroid).
        direction (np.ndarray): A unit direction vector of the best-fit line.
    """
    # Ensure input is a numpy array
    points = np.asarray(points)

    # Compute the centroid (a point on the best-fit line)
    centroid = points.mean(axis=0)

    # Subtract the centroid to center the data
    centered_points = points - centroid

    # Compute the covariance matrix of centered points
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Get the eigenvector with the largest eigenvalue (direction of max variance)
    direction = eigenvectors[:, np.argmax(eigenvalues)]

    return centroid, direction

def find_tracks(file, min_event_nhit=40, use_prompt=False, exclude_disabled=True):

    # initialize data dictionary
    data_store = {}
    for key in data_store_nominal.keys(): data_store[key] = []

    # open file data/references
    f = h5py.File(file)
    events = f['charge/events/data']['id'][ f['charge/events/data']['nhit']>min_event_nhit]
    events_hits_ref = f['charge/events/ref/charge/calib_final_hits/ref'][:]
    events_hits_rref = f['charge/events/ref/charge/calib_final_hits/ref_region'][:]
    hit_data = f['charge/calib_final_hits/data'][:]

    events_trig_rref = np.array(f['charge/raw_events/ref/charge/ext_trigs/ref'][:])
    ext_trig_data = f['charge/ext_trigs/data'][:]

    #

    if use_prompt:
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
        
        for io_group in range(1,9):
            
            pixel_pitch = geo[io_group]['pixel_pitch']
            module = (io_group-1)//2
            anode = (io_group+1) % 2 + 1
            
            hits = ev_hits.copy()
            
            mog = hits['io_group'].astype(int) == io_group
            #require at least 20 hits total in this io group
            #if all hits in a line, ~8cm of track
            if np.sum(mog)<20: continue
                
            _X = np.array([hits['x'][mog], hits['y'][mog], hits['z'][mog]]).transpose()
            _Q = np.array(hits['Q'][mog])
            all_hit_io_group = np.array(hits['io_group'][:][mog])
            all_hit_io_chan = np.array(hits['io_channel'][:][mog])
            all_hit_chip_id = np.array(hits['chip_id'][:][mog])
            all_hit_chan_id = np.array(hits['channel_id'][:][mog])

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

                #do PCA
                pca=PCA(n_components=3)
                pca.fit(X)

                #make sure this subsection is track-like
                if np.max(pca.explained_variance_ratio_) < 0.98: 
                    continue
                    
                # fit subcluster to a line
                c, ssubdir = fit_line_3d(X)

                #get angular variance between section and original fit
                cos_theta = np.dot(ssubdir, fdirection) / np.linalg.norm(ssubdir) / np.linalg.norm(fdirection)

                # check that this cluster is aligned with original track direction
                if np.absolute(cos_theta) < 0.98: 
                    continue


                # squared distance of all points from line
                # get distance from each point to center
                dist_from_cent = X - c
                norm = np.sqrt(np.sum( dist_from_cent**2, axis=1 ))

                #project onto direction of track
                projection_on_direction = dist_from_cent[:,0]*ssubdir[0] \
                                        + dist_from_cent[:,1]*ssubdir[1] \
                                        + dist_from_cent[:,2]*ssubdir[2]

                perp_component = dist_from_cent - np.array([ projection_on_direction * ssubdir[0], projection_on_direction * ssubdir[1], projection_on_direction * ssubdir[2] ]  ).transpose()
                distance_perp = np.sqrt(np.sum(perp_component**2, axis=1))

                # check that mean of best 90% of hits are within 1 pixel pitch from fit
                if scipy.stats.trim_mean( distance_perp, 0.05 ) > pixel_pitch: continue


                # remove hits that are more than 2 pixel pitchs from fit
                filt_mask = distance_perp < pixel_pitch*2
                
                X_filt = X[filt_mask]
                Q_filt = Q[filt_mask]
                hit_io_group_filt = hit_io_group[filt_mask]
                hit_io_chan_filt =  hit_io_chan[filt_mask]
                hit_chip_id_filt = hit_chip_id[filt_mask]
                hit_chan_id_filt = hit_chan_id[filt_mask]

                #re-do the fit without the stragglers!
                centr, direction = fit_line_3d(X_filt)

                #get track angle to drift direction
                angle = np.arccos( np.absolute(direction[0]) / np.linalg.norm(direction))

                #get track angle in yz plane:
                angle_yz = np.arctan(np.absolute(direction[2])/np.absolute(direction[1]))

                #remove tracks that are exactly vertical or exactly horizontal, in case these are some funny artifacts
                if (np.absolute(angle_yz) < 0.01) or (np.absolute(angle_yz - np.pi/2) < 0.01): continue

                # get predicted pixels which should have hits
                norm = np.sqrt(direction[1]**2 + direction[2]**2)
                
                #get distance along track from center ignoring drift direction
                projection_on_direction = dist_from_cent[:,1]*direction[1]/norm \
                                    + dist_from_cent[:,2]*direction[2]/norm \

                # length in pixel plane only
                length = np.max(projection_on_direction) - np.min(projection_on_direction) 

                if length < 12*pixel_pitch: continue

                length=length-8*pixel_pitch

                # now, get all hit pixels in forward direction
                start=centr - direction/norm * length/2
                end = centr + direction/norm * length/2

                # improve pixel-hit predictions by allowing for kinks in the track
                # check for a peak in the fit residual distribution

                starts = [start]
                ends   = [end]

                kink_depth=2

                def add_kinks(start, end, pts, qs):

                    c = (start+end)/2
                    direction = (end-start)/np.linalg.norm(end-start)

                    #recalculate perpendicular distances
                    dist_from_cent = pts - c
                    norm = np.sqrt(np.sum( dist_from_cent**2, axis=1 ))

                    #project onto direction of track
                    projection_on_direction = dist_from_cent[:,0]*direction[0] \
                                            + dist_from_cent[:,1]*direction[1] \
                                            + dist_from_cent[:,2]*direction[2]

                    perp_component = dist_from_cent - np.array([ projection_on_direction * direction[0], projection_on_direction * direction[1], projection_on_direction * direction[2] ]  ).transpose()
                    distance_perp = np.sqrt(np.sum(perp_component**2, axis=1))

                    # get pt with max perpendicular distance, find charge centroid within sqrt(2) pixel pitch sphere of that pt
                    max_pt_indx = np.argmax(distance_perp)
                    max_pt_xyz = pts[max_pt_indx, :]

                    close_pts_mask = np.sum((pts - max_pt_xyz)**2, axis=0) < np.sqrt(2)*pixel_pitch

                    centroid =  np.array([np.sum(pts[:,0]*qs), np.sum(pts[:,1]*qs), np.sum(pts[:,2]*qs)]) / np.sum(qs)

                    return centroid
                    
        
                ##################################################################
                
                for ikink in range(kink_depth):

                    new_starts = deepcopy(starts)
                    new_ends   = deepcopy(ends)

                    add_pts = 0

                    for i in range(len(starts)):

                        start = starts[i]
                        end = ends[i]

                        pts_mask = ( X_filt[:,1] > start[1]-pixel_pitch ) & ( X_filt[:,2] > start[2]-pixel_pitch ) & ( X_filt[:,1] < end[1]+pixel_pitch ) & ( X_filt[:,2] < end[2]-pixel_pitch ) 

                        if np.sum(pts_mask)<20: continue
                            
                        new_pt = add_kinks( start, end, X_filt[pts_mask, :], Q_filt[pts_mask] )

                        if (np.linalg.norm( start - new_pt ) > 2*pixel_pitch) and (np.linalg.norm( end - new_pt ) > 2*pixel_pitch):
                            new_starts.insert(i+1+add_pts, new_pt)
                            new_ends.insert(i+add_pts, new_pt)
                            add_pts+=1

                    starts = new_starts
                    ends = new_ends

                # get all pts which would be hit by these track segments
                pyz, l = get_all_hit_pixels_kinked( starts, ends, io_group)

                if l.shape[0]==0:
                    continue

                ###############################################

                

                # at this point, we have a prediction for pixels which should have been hit by this track.
                # Now, if using prompt hits is desired, we can swap these in!

                #SWAP IN PROMPT HITS if use_prompt!
                ##################################
                if use_prompt:
                    hits = dereference.dereference(ev, events_phits_ref, phit_data, region=events_phits_rref).flatten()
                    if exclude_disabled: hits = hits[ ~(hits['is_disabled'][:])  ]
                    mog = hits['io_group'].astype(int)== io_group
                    X = np.array([hits['x'][mog], hits['y'][mog], hits['z'][mog]]).transpose()
                    Q = np.array(hits['Q'][mog])
                    hit_io_group = np.array(hits['io_group'][:][mog])
                    hit_io_chan = np.array(hits['io_channel'][:][mog])
                    hit_chip_id = np.array(hits['chip_id'][:][mog])
                    hit_chan_id = np.array(hits['channel_id'][:][mog])

                    # now filter out hits that are too far from this track (>1 pixel pitch)
            
                    #get distance of hits to the line between start, end
                    track_dir = (direction)/np.linalg.norm(direction)
                    p=(X-centr)
                    dp=( p[:, 0]*track_dir[0] +  p[:, 1]*track_dir[1] +  p[:, 2]*track_dir[2]  )
                    
                    approach_to_track = p - np.array([dp*track_dir[0], dp*track_dir[1], dp*track_dir[2]]).transpose()

                    dist_to_track = np.linalg.norm(approach_to_track, axis=1)
                    #print(dist_to_track.shape, p.shape)
                    
                    filt_mask = dist_to_track < 2*pixel_pitch

                    X_filt = X[filt_mask]
                    Q_filt = Q[filt_mask]
                    hit_io_group_filt = hit_io_group[filt_mask]
                    hit_io_chan_filt =  hit_io_chan[filt_mask]
                    hit_chip_id_filt = hit_chip_id[filt_mask]
                    hit_chan_id_filt = hit_chan_id[filt_mask]
                    
                ##################################

                    

                # find which pixels were MISSED
                all_hit_pixel_ids = get_unique_pixel_ids_yz( X_filt[:,1:], io_group )
                hit_pixel_ids, hit_idx, hit_counts = np.unique(all_hit_pixel_ids, return_index=True, return_counts=True )
            
                predicted_pixel_ids = get_unique_pixel_ids_yz(  pyz, io_group  )
                #Now, save summary information, as well as pixel-by-pixel information

                data_store['angle']+=[angle]*l.shape[0]
                data_store['module']+=[module]*l.shape[0]
                data_store['pixel_id'] += list(get_unique_pixel_ids_yz( pyz, io_group ))
                data_store['pixel_y'] += list( pyz[:,0] )
                data_store['pixel_z'] += list( pyz[:,1] )
                data_store['pixel_length'] += list(l)
                data_store['anode'] += [anode]*l.shape[0]
                data_store['trigger_tag'] += [trig_tag]*l.shape[0]

                N=len(get_unique_pixel_ids_yz( pyz, io_group ))

                _total_charge_collected = [-1]*N
                _hit_drift = [-1]*N
                _is_hit = [False]*N
                _hit_io_group = [-1]*N
                _hit_io_chan = [-1]*N
                _hit_chip_id = [-1]*N
                _hit_chan_id = [-1]*N
                
                for ipid, pid in enumerate(list( get_unique_pixel_ids_yz( pyz, io_group ) )):
                    if pid in hit_pixel_ids:
                        indxs = np.where(all_hit_pixel_ids==pid)[0]
                        _total_charge_collected[ipid] =  np.sum(Q_filt[indxs]) 
                        _hit_drift[ipid] = np.mean(X_filt[:,0][indxs]) 
                        _is_hit[ipid] = True
                        _hit_io_group[ipid] = hit_io_group_filt[indxs[0]] 
                        _hit_io_chan[ipid] = hit_io_chan_filt[indxs[0]]
                        _hit_chip_id[ipid] = hit_chip_id_filt[indxs[0]]
                        _hit_chan_id[ipid] = hit_chan_id_filt[indxs[0]]

                data_store['total_charge_collected'] += _total_charge_collected
                data_store['hit_drift'] += _hit_drift
                data_store['is_hit'] += _is_hit
                data_store['hit_io_group'] += _hit_io_group
                data_store['hit_io_chan'] += _hit_io_chan
                data_store['hit_chip_id'] += _hit_chip_id
                data_store['hit_chan_id'] += _hit_chan_id
                 
                if not len(data_store['pixel_id'])==len(data_store['total_charge_collected']):
                    print('wot wot wot???')                                           
                    
    return data_store

def main(data_dir, outf_name, use_prompt, max_files, specify_run=None):
    
    create_h5_file_safe(outf_name)

    data_files = sorted(os.listdir(data_dir))
    close_all_h5_files()

    files_to_parse = len(data_files)
    if max_files > 0:
        files_to_parse = max_files
    if True:
        for ifile in tqdm(range(files_to_parse), desc='parsing files...'):
            
            # do main file processing
            if not specify_run is None:
                if not str(specify_run) in nominal_files[ifile]: continue

            dstore = find_tracks(data_dir + data_files[ifile], use_prompt=use_prompt) 

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
        parser.add_argument('--outf_name', '-o', default='default_outf.h5', type=str, help='''Name/path for output to be written''')
        parser.add_argument('--use_prompt', action='store_true', help='''Use prompt hits to study efficiency''')
        parser.add_argument('--max_files', default=-1, type=int, help='''Max number of files to process''')
        args = parser.parse_args()
        main(**vars(args))
