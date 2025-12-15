import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, os, sys, glob
from tqdm import tqdm
import xarray as xr
from multiprocessing import Pool

## download files not downloaded on first pass through download_mt.py
## need to run count_files.py first

# start and end date
## set in count_files.py

# buffer time for preprocessing 
buffer_time = pd.Timedelta(1, 'd')
download_dir = '/pscratch/sd/d/dcarrel/ir_volcano_mt3'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)
    
# query start date

num_files = xr.open_dataset('diff_days.nc')['file_diff'].load()
dq = num_files.time.values
num_files = num_files.values

num_processes = 32
total_files = np.sum(num_files)
files_per_process = np.ceil(total_files/num_processes)
cum_files = np.floor((np.cumsum(num_files) / files_per_process))

days2download = [dq[cum_files == i] for i in range(num_processes)]

def download_images_daily(day):
    qstart = pd.to_datetime(day)
    qend = day + pd.Timedelta(1,'d')
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

    already_downloaded = glob.glob(download_dir+'/'+qstart.strftime('%Y-%m-%d')+'*')
    timestamps = sorted([f.split('/')[-1].split('.')[0] for f in already_downloaded])
    for image in tqdm(images):
        
        if image['timestamp'] in timestamps:
            continue

        image_response = requests.get(
            f"https://avert-legacy.ldeo.columbia.edu/api/imagery/infrared/r/{image['image_id']}",
            params={"download": True}
        )
        time = pd.to_datetime(image['timestamp'])
        tstr = time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(f"{download_dir}/{tstr}.jpg", 'wb') as f:
            f.write(image_response.content)

def download_days(days):
    for day in days:
        download_images_daily(day)


if __name__=='__main__':
    pool = Pool(processes=num_processes)
    pool.map(download_days, days2download)