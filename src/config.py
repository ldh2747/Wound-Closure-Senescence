"""Central path configuration. Set the environment variable MIDAS_DATA_ROOT
(or edit DATA_ROOT below) to the folder that holds the raw TIFF images.

Expected layout (names as in the original acquisition folders):

    <DATA_ROOT>/
        p5/{0h,24h,48h}/<plate folder>/<position>.tif                # Experiment 1, P5
        p15/26.09.29 hdf p 15 migration/{0h,24h,48h}/<plate>/<pos>.tif   # Experiment 1, P15
        20261013,4 AI hDF P5,15 migration(additional_data)/              # Experiment 2
            20261013 AI hDF P5 migration/{0h,24h,48h}/...
            20261014 AI hdf P15 migrtion/{0h,24h,48h}/...
        work/                                                         # outputs (caches, csv, models)
"""
import os

DATA_ROOT = os.environ.get("MIDAS_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_root"))
BASE = DATA_ROOT
WORK = os.path.join(BASE, "work")
ADD = os.path.join(BASE, "20261013,4 AI hDF P5,15 migration(additional_data)")
os.makedirs(WORK, exist_ok=True)
