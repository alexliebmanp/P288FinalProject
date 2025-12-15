import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, os, sys
import xarray as xr
from tqdm import tqdm
from multiprocessing import Pool


### Function: query images from AVERT API
### and count num of available image files per day
### to inform downloading scheme
###

# start and end date
start_date = pd.to_datetime('2024-01-31 00:00:00')
end_date = pd.to_datetime('2025-12-09 00:00:00')

# buffer time for preprocessing 
buffer_time = pd.Timedelta(1, 'd')
download_dir = '/pscratch/sd/d/dcarrel/ir_volcano_mt2'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
    
# query start date
qstart = start_date - buffer_time
qend = end_date + buffer_time
## want to get number of images as a function of time
dq = pd.date_range(qstart,qend, freq='1d')
dqs, dqe = dq[:-1], dq[1:]
num_files = np.zeros_like(dq, dtype='int')
for i, (s,e) in tqdm(enumerate(zip(dqs,dqe))):
    startstr = s.strftime('%Y-%m-%dT%H:%M')
    endstr = e.strftime('%Y-%m-%dT%H:%M')
    
    
    response = requests.get(
        "https://avert-legacy.ldeo.columbia.edu/api/imagery/infrared/query",
        params={
            "site": "VPMI",
            "vnum": 345040,
            "search_from": startstr,
            "search_to": endstr
        })
    images = response.json()
    num_files[i] = len(images)

xr.Dataset({'num_files': xr.DataArray(num_files, coords={'time':dq})}).to_netcdf('num_files.nc')