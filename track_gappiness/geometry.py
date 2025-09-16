import json
import numpy as np


def load_geo(filename='geo-2x2.json'):
    ''' Load geometry file and np-ize it'''
    geo_json={}
    with open(filename, 'r') as f:
        geo_json = json.load(f)

    geo = {}
    for io_group in geo_json.keys():
        y, z = geo_json[io_group]['pix_y_z']
        geo[int(io_group)] = {
            'pix_y_z' :  ( np.array(y), np.array(z) ),
            'pixel_pitch' : geo_json[io_group]['pixel_pitch'],
            'min_y_z' : geo_json[io_group]['min_y_z']
        }

    return geo


def save_geo(geo_dict, filename='geo-2x2.json'):
    '''Save geometry from arrays dictionary with io_group : pixel y, pixel z'''
    
    geo = {}
    for io_group in geo_dict.keys():
        y, z = geo_dict[io_group]
        geo[int(io_group)] = {
            'pix_y_z' :  ( list(y), list(z) ),
            'pixel_pitch' : np.mean( np.diff(y)[5:-5] ),
            'min_y_z' : ( np.min(y), np.min(z) )
        }

    with open(filename, 'w') as f:
        json.dump(geo, f, indent=4)

    return filename

                

