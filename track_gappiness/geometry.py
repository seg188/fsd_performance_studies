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



