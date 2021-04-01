import argparse
import sys
import time
from io import BytesIO

import h5py
import lmdb
import numpy as np
from PIL import Image

from storage import STORAGE_TYPES

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, help="The database to store extracted frames")
    parser.add_argument("--db_type", type=str, choices=["LMDB", "HDF5", "FILE", "PKL"], default="LMDB",
                        help="Type of the database")
    args = parser.parse_args()

    start = time.time()
    meta, img = STORAGE_TYPES[args.db_type].info(args.db_name)
    print(meta)
    if img is not None:
        img.show()
    print('Time:', time.time() - start)



