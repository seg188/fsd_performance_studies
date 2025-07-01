import h5py
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import os
from numpy import ma
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import datetime
from datetime import tzinfo

from dereference import dereference, get_event_hits_packets

def set_axes_equal(ax):
    return
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlicm3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    
bounds =[[-67, 67.], [-67, 67.], [-67.,  67.]]

def plot_geo(ax, bounds):
    
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_zlim(bounds[2][0], bounds[2][1])
    
    ax.set_xlabel('z [cm]', fontsize=14, weight='bold')
    ax.set_ylabel('y [cm]', fontsize=14, weight='bold')
    ax.set_zlabel('x [cm]', fontsize=14, weight='bold')
    
    # top
    x = [bounds[0][0],bounds[0][1],bounds[0][1],bounds[0][0]]
    y = [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]]
    z = [bounds[2][1],bounds[2][1],bounds[2][1],bounds[2][1]]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.08, color='grey'))
    
    # bottom
    x = [bounds[0][0],bounds[0][1],bounds[0][1],bounds[0][0]]
    y = [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]]
    z = [bounds[2][0],bounds[2][0],bounds[2][0],bounds[2][0]]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.08, color='grey'))
    
    # left
    x = [bounds[0][0],bounds[0][0],bounds[0][0],bounds[0][0]]
    y = [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]]
    z = [bounds[2][1],bounds[2][0],bounds[2][0],bounds[2][1]]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.08, color='grey'))
    
    # right
    x = [bounds[0][1],bounds[0][1],bounds[0][1],bounds[0][1]]
    y = [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]]
    z = [bounds[2][1],bounds[2][0],bounds[2][0],bounds[2][1]]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.08, color='grey'))
    
    # front
    x = [bounds[0][0],bounds[0][0],bounds[0][1],bounds[0][1]]
    y = [bounds[1][0],bounds[1][0],bounds[1][0],bounds[1][0]]
    z = [bounds[2][0],bounds[2][1],bounds[2][1],bounds[2][0]]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.08, color='grey'))
    
    # back
    x = [bounds[0][0],bounds[0][0],bounds[0][1],bounds[0][1]]
    y = [bounds[1][1],bounds[1][1],bounds[1][1],bounds[1][1]]
    z = [bounds[2][0],bounds[2][1],bounds[2][1],bounds[2][0]]
    verts = [list(zip(x,y,z))]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.08, color='grey'))

def compare_prompt_final(prompt, final, view_init=None):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    sc=ax.scatter( prompt['z'].flatten(), prompt['y'].flatten(), prompt['x'].flatten(), c='red', marker='.', s=0.4, alpha=0.7, cmap='plasma' )
    sc=ax.scatter( final['z'].flatten(), final['y'].flatten(), final['x'].flatten(), c='green', marker='.', s=0.4, alpha=0.7, cmap='plasma' )
    
    plot_geo(ax, bounds)
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    if not view_init is None: ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.show()
    return ax
def plot_cathode(ax):
    # Define the plane's equation (ax + by + cz + d = 0)
    a = 0
    b = 0
    c = 1
    d = 0

    # Create a grid of points on the plane
    x = np.linspace(bounds[2][0], bounds[2][1], 10)
    y = np.linspace(bounds[1][0], bounds[1][1], 10)
    X, Y = np.meshgrid(x, y)
    Z = (-a*X - b*Y - d) 

    # Plot the plane
    ax.plot_surface(Z, Y, X, alpha=0.1, color='blue')
    
    
def display(ff, event=0, fname='Sample File'):
    
    hits, packets = get_event_hits_packets(ff, event)
    
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    sc=ax.scatter( hits['z'].flatten(), hits['y'].flatten(), hits['x'].flatten(), c=hits['Q'].flatten(), marker='.', s=0.2, alpha=0.7 )
    
    if type(event)==int:
        events = ff['charge/events/data']
        ax.set_title('Event {}\n{}\n{}'.format(event, fname, datetime.datetime.fromtimestamp(events[event]['unix_ts']+120*60)))
    else:
        ax.set_title('Event Collection\n{}'.format(fname))
    plot_geo(ax, bounds)
    #ax.view_init(elev=180, azim=20)
    cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    
    plt.show()

    
from matplotlib import colors as mcolors

def display_final_prompt(fhits, hits, fname='Sample File', labels=None):
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    cc=0.25
    clusters = np.ones( fhits.shape[0]+hits.shape[0] ).astype(int)
    clusters[ 0:fhits.shape[0] ] = 2
                                                            
    cs=np.array([colors[clust] if clust >-10 else 'grey' for clust in clusters])

    sc=ax.scatter( np.concatenate( (fhits['z'].flatten(),hits['z'].flatten()) ), 
                   np.concatenate( (fhits['y'].flatten(),hits['y'].flatten()) ),  
                   np.concatenate( (fhits['x'].flatten(),hits['x'].flatten()) ), 
                  c=cs, marker='.', s=1, alpha=cc )
        
    plot_geo(ax, bounds)
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    plt.show()
    return ax

def display_cluster_legend(hits, clusters, fname='Sample File', labels=None):
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    cc=0.6
    cs=np.array([colors[clust] if clust >-10 else 'grey' for clust in clusters])
    
    for l in set(clusters):
        mask =clusters==l
        if l==0:
            label='ignored'
        else:
            label='used in fit'
        sc=ax.scatter( hits[mask]['z'].flatten(), hits[mask]['y'].flatten(), hits[mask]['x'].flatten(), label=label, c=cs[mask], marker='.', s=1, alpha=cc )
        
    ax.legend(markerscale=8, fontsize=12)
        
    plot_geo(ax, bounds)
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    plt.show()
    return ax

def display_cluster(hits, clusters, fname='Sample File', labels=None):
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    cc=0.25
    cs=np.array([colors[clust] if clust >-10 else 'grey' for clust in clusters])

    sc=ax.scatter( hits['z'].flatten(), hits['y'].flatten(), hits['x'].flatten(), c=cs, marker='.', s=1, alpha=cc )
        
    plot_geo(ax, bounds)
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    plt.show()
    return ax
    
    
def display_hits(hits, fname='Sample File', view_init=None):
    
    fig=plt.figure(figsize=(8,12))
    ax=fig.add_subplot(projection='3d')
    sc=ax.scatter( hits['z'].flatten(), hits['y'].flatten(), hits['x'].flatten(), c=np.log(hits['Q']), marker='.', s=0.4, alpha=0.7, cmap='plasma' )
    
    plot_geo(ax, bounds)
    plot_cathode(ax)
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    if not view_init is None: ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])
    set_axes_equal(ax)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    plt.show()

    return ax

def display_hits_with_colorscale(hits, colorscale, fname='Sample File', view_init=None):
    
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    sc=ax.scatter( hits['z'].flatten(), hits['y'].flatten(), hits['x'].flatten(), c=colorscale, marker='.', s=0.4, alpha=0.7, cmap='plasma' )
    
    plot_geo(ax, bounds)
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    if not view_init is None: ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.show()
    return ax

def display_hits_with_fit(hits, point, line, fname='Sample File', view_init=None):
    
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(projection='3d')
    sc=ax.scatter( hits['z'].flatten(), hits['y'].flatten(), hits['x'].flatten(), c=np.log(hits['Q']), marker='.', s=0.4, alpha=0.7, cmap='plasma' )
    
    plt_pts=np.linspace(-50, 50, 1000)
    pts = np.array( [point + plt_pt * line for plt_pt in plt_pts ] )
    ax.scatter(pts[:,2], pts[:,1], pts[:,0], s=0.2, marker='_', color='black')
    
    plot_geo(ax, bounds)
    
    
    #ax.view_init(elev=180, azim=20)
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    if not view_init is None: ax.view_init(elev=view_init[0], azim=view_init[1])
    
    
    plt.show()
    return ax
    
def generate_legend():
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    colors=['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'violet', 'pink' ]
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot()
    for i in range(8):
        sc=ax.scatter( range(1,8), range(1,8), c=colors[i], marker='.', s=100, label='io_group={}'.format(i+1) )
    lgd=ax.legend(facecolor='white', fontsize=12)
    
def display_filter(hits, clusters, fname='Sample File', view_init=None):
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    fig=plt.figure(figsize=(9,9))
    ax=fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('Unfiltered')
    mask = clusters<2
    nmask = np.logical_not(mask)
    sc=ax.scatter( hits['z'][mask].flatten(), hits['y'][mask].flatten(), hits['x'][mask].flatten(), c='green', marker='.', s=0.2, alpha=0.7, label='remaining hits' )
    sc=ax.scatter( hits['z'][nmask].flatten(), hits['y'][nmask].flatten(), hits['x'][nmask].flatten(), c='red', marker='.', s=0.2, alpha=0.7, label='removed' )
    plot_geo(ax, bounds)
    if not view_init is None: ax.view_init(elev=view_init[0], azim=view_init[1])
    lgd=ax.legend(fontsize=10)
    for legend_handle in lgd.legend_handles:
        legend_handle.set_sizes([50])
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    
    ax=fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('Filtered')
    mask = clusters<2
    sc=ax.scatter( hits['z'][mask].flatten(), hits['y'][mask].flatten(), hits['x'][mask].flatten(), c=hits['Q'][mask], marker='.', s=0.2, alpha=0.7 )
    plot_geo(ax, bounds)
    if not view_init is None: ax.view_init(elev=view_init[0], azim=view_init[1])
    #cbar=plt.colorbar(sc, ax=ax, fraction=0.05)
    #cbar.set_label(label='charge [ke-]',weight='bold', fontsize=14, rotation=-90, labelpad=10)
    plt.show()
    


    
