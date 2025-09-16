#DATA FORMAT WRITTEN TO OUTPUT FILE WITH DESCRIPTIONS
# 
# Every track that projects onto a pixel will have 1 entry PER PIXEL CROSSED in the format below
# - Each list will always have the same length, and an entry at index j is associated with a unique track+pixel
data_store_nominal = {
        #unique track counter, a unique integer for each track/event used
        'unique_track_id' : [],
        # pixel id crossed by a track section (defined below)
        'pixel_id' : [], 
        # pixel y position crossed by a track section
        'pixel_y' : [],
        # pixel z position crossed by a track section
        'pixel_z' : [],
        # angle of track crossing pixel in x,z plane
        'angle'    : [],
        #which module of 2x2 the hits were in. Always 1 for FSD.
        'module' : [], 
        # (1,2) for which drift region the track was in
        'anode' : [],
        # total length of the track crossing 
        'total_length' : [],
        # length of the track crossing this particular pixel
        'pixel_length' : [],
        # bool - found a hit consistent with this track crossing 
        'is_hit' : [],
        # drift time of hit if found, else -1
        'hit_drift' : [],
        # io_group of hit if found, else -1
        'hit_io_group' : [],
        # io_chan of hit if found, else -1
        'hit_io_chan' : [],
        # chip_id of hit if found, else -1
        'hit_chip_id' : [],
        # chan_id of hit if found, else -1
        'hit_chan_id' : [],
        # charge sum of all hits on this channel associated with this track
        'total_charge_collected' : [],
        # tag for which io group received external trigger (or -1 for none)
        'trigger_tag' : [],
        # distance of point to track fit
        'approach_length' : [],
        # pt of closest apprach (y) on track
        'approach_pt_y' : [],
        # pt of closest apprach (z) on track
        'approach_pt_z' : [],
        # pixel closest to pt of closest apprach (y) on track
        'pix_approach_pt_y' : [],
        # pixel closest to pt of closest apprach (z) on track
        'pix_approach_pt_z' : [],
        }
#


