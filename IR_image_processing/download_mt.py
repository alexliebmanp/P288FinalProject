import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, os, sys
from tqdm import tqdm
import xarray as xr
from multiprocessing import Pool

## use: download IR images quickly

# start and end date
## set in count_files.py

# buffer time for preprocessing 
buffer_time = pd.Timedelta(1, 'd')
download_dir = '/pscratch/sd/d/dcarrel/ir_volcano_mt3'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
    
# query start date

num_files = xr.open_dataset('num_files.nc')['num_files'].load()
dq = num_files.time.values
num_files = num_files.values

num_processes = 64
total_files = np.sum(num_files)
files_per_process = np.ceil(total_files/num_processes)
cum_files = np.floor((np.cumsum(num_files) / files_per_process))
cum_files = xr.DataArray(cum_files, coords={'time': dq})

dates = np.array([[0,0]]*num_processes, dtype='datetime64[m]')

for i in range(num_processes):
    start_time = cum_files.where(cum_files == i, drop=True)
    dates[i][0] = start_time.time.min().values
    dates[i][1] = start_time.time.max().values
    dates[i][1] -= pd.Timedelta(1,'s')

def download_images(start_end_times):
    qstart, qend = start_end_times
    qstart = pd.to_datetime(qstart)
    qend = pd.to_datetime(qend)
    startstr = qstart.strftime('%Y-%m-%dT%H:%M')
    endstr = qend.strftime('%Y-%m-%dT%H:%M')

    response = requests.get(
        "https://avert-legacy.ldeo.columbia.edu/api/imagery/infrared/query",
        params={
            "site": "VPMI",
            "vnum": 345040,
            "search_from": startstr,
            "search_to": endstr
        }
    )
    images = response.json()
    
    for image in tqdm(images):
        image_response = requests.get(
            f"https://avert-legacy.ldeo.columbia.edu/api/imagery/infrared/r/{image['image_id']}",
            params={"download": True}
        )
        time = pd.to_datetime(image['timestamp'])
        tstr = time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(f"{download_dir}/{tstr}.jpg", 'wb') as f:
            f.write(image_response.content)

if __name__=='__main__':
    pool = Pool(processes=num_processes)
    pool.map(download_images, dates)