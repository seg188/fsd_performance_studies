import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy


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

    return centroid, direction/np.linalg.norm(direction)
    

def get_perp_and_para_components(X, center, direction):

    dist_from_cent = X - center
    norm = np.sqrt(np.sum( dist_from_cent**2, axis=1 ))

    #project onto direction of track
    projection_on_direction = dist_from_cent[:,0]*direction[0] \
                            + dist_from_cent[:,1]*direction[1] \
                            + dist_from_cent[:,2]*direction[2]

    perp_component = dist_from_cent\
                        - np.array([\
                        projection_on_direction * direction[0],\
                        projection_on_direction * direction[1],\
                        projection_on_direction * direction[2] ]\
                              ).transpose()
    
    distance_perp = np.sqrt(np.sum(perp_component**2, axis=1))


    return projection_on_direction, distance_perp

    

def fit_tracks(X,\
               max_segment_length=10.0,\
               full_event_pca_variance_ratio=0.98,\
               min_segment_angular_variation_cos=0.9,
               trim_track_ends=2.,
               trim_mean_fraction=0.1,
               trim_mean_cut=0.5,
               prune_distance=1.0,
               min_subfit_nhit=10,
               subseg_max_pca_variance=0.99,
               sum_sqs_cut=0.2):

    # FIND straight track segments with high confidence
    # returns:
    # - start and end points of track fits, mask of which hits were used/pruned

    # do basic PCA to ensure all of the points are roughly linear
    pca=PCA(n_components=3)
    pca.fit(X)

    #make sure these hits are roughly track-like
    if np.max(pca.explained_variance_ratio_) < full_event_pca_variance_ratio: 
        return None, None

    #fit cluster to a line and assign a length along track direction

    # fit subcluster to a line
    center, direction = fit_line_3d(X)

    # Get distance projected on track direction for each point
    # get distance from each point to center
    
    para, perp = get_perp_and_para_components(X, center, direction)

    # use np.digitize to break into bins of size max_segment_length

    total_length = np.max( para ) - np.min( para ) - 2*trim_track_ends
    
    n_bins = int(total_length // max_segment_length) + 1 
    
    bins = np.linspace( np.min( para ) + trim_track_ends, np.max( para ) - trim_track_ends, n_bins)
        
    indices = np.digitize( para, bins=bins )

    # for each bin index, perform the linear fit.
    index_fit_results = {}
    
    for index in set(indices):
        mask = indices==index

        if np.sum(mask)<min_subfit_nhit: continue

        _X = X[mask]

        c, d = fit_line_3d(_X)

        # if the direction is too far off from main fit, throw away this info
        if np.dot(d, direction) / ( np.linalg.norm(d) * np.linalg.norm(direction) ) < min_segment_angular_variation_cos:
            continue
        
        _para, _perp = get_perp_and_para_components(_X, c, d)


        if scipy.stats.trim_mean( _perp, trim_mean_fraction ) > trim_mean_cut:
            continue

        # trim hits which are too far from the fit
        _mask = _perp < prune_distance

        # re-do the fit without the stragglers
        c, d = fit_line_3d(_X[_mask])
        _para, _perp = get_perp_and_para_components(_X[_mask], c, d)
        sum_sqs = np.sqrt(np.sum( _perp**2 ))/_perp.shape[0]

        if sum_sqs > sum_sqs_cut:
            continue
        
        _length = np.max(_para) - np.min(_para)

        pca_max_variance = np.max(PCA(n_components=3).fit(_X).explained_variance_ratio_)

        if pca_max_variance < subseg_max_pca_variance:
            continue
        
        index_fit_results[index] = {
            'center'    : c,
            'direction' : d,
            'start'     : c - _length/2*d,
            'end'       : c + _length/2*d,
            'sum_sqs'   : sum_sqs,
            'pca'       : pca_max_variance,
            'angle'     : np.arccos( np.absolute(d[0]) / np.linalg.norm(d))    
        }        

    return index_fit_results, indices




    