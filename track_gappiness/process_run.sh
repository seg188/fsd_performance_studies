#!/bin/bash


#PROCESS A RUN

module load python

RUN=$1
OUTFNAME=/pscratch/sd/s/sgberg/fsd_track_efficiency_data/scan_12152025/fsd_run_${RUN}_reflow_v7_v3.h5

echo $OUTFNAME

# RUN THE FIRST STAGE TRACK EFFICIENCY STUDY
python run_track_efficiency_study.py --data_dir /global/cfs/cdirs/dune/www/data/FSD/reflows/v7/flow/cosmics/config_scan/ --outf_name $OUTFNAME  --version_tag fsd --specify_run $RUN


# RUN THE DQDX STUDY ON THE RESULTS
OUTFNAME_DQDX=/pscratch/sd/s/sgberg/fsd_track_efficiency_data/scan_12152025_copy/fsd_run_${RUN}_dqdx_reflow_v7_v3.h5

python run_dqdx_study.py -i $OUTFNAME -o $OUTFNAME_DQDX -t fsd
