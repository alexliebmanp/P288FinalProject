import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import imread
import os, glob
import xarray as xr
from tqdm import tqdm
import shutil
from multiprocessing import Pool

scratch_dir = '/pscratch/sd/d/dcarrel/'
download_dir = '/pscratch/sd/d/dcarrel/ir_volcano_mt3'

files1 = glob.glob(scratch_dir+'/poas_bad/*.jpg')
files2 = glob.glob(scratch_dir+'/poas_good/*.jpg')
files = sorted(files1+files2)
num_files = len(files)
num_processes = 128
processes_per_file = np.ceil(num_files/num_processes)
subfiles = np.array_split(np.array(files, dtype='str'), 32)

def invert_mask(mask):
    mask = np.where(mask == 0, 1, np.nan)
    return mask

mask1 = np.load('fm1.npy')
mask1 = invert_mask(mask1)
t1 = pd.to_datetime('2025-04-03')
mask2 = np.load('fm2.npy')
mask2 = invert_mask(mask2)
t2 = pd.to_datetime('2025-08-20')
mask3 = np.load('fm3.npy')
mask3 = invert_mask(mask3)

## gets metrics FOR ELIMINATION for given file
def get_met(f, datetime):
    mask=mask1
    
    if datetime > t1 and datetime < t2:
        mask=mask2
    elif datetime > t2:
        mask=mask3

    xlb, xub = 200, 400 ## where the lake is expected to be
    ylb, yub = 0, 400

    im = imread(f)
    above_lake = im[:200,:400]
    over_lake = (mask*im)[200:400, :400]


    mean_above_lake = np.nanmean(above_lake)
    mean_lake = np.nanmean(over_lake)
    max_lake = np.nanmax(over_lake)
    
    upper_mean = np.nanmean(above_lake)
    
    xmean = np.nanmean(im,axis=0)
    xnorm = np.abs(np.nanmean(xmean[:3])-np.nanmean(im))
    
    return xnorm, max_lake, np.abs(upper_mean - mean_lake)

def filter_images(files2filter):
    for f in tqdm(files2filter):
        fname = f.split('/')[-1]
        datetime = pd.to_datetime(fname.split('.')[0])
        
        cant_load=False

        ## check if dates are near when camera changes frame 
        first_frame_change = datetime.strftime('%Y%m%d') == '20250402'
        ti = pd.to_datetime('2025-08-17')
        tf = pd.to_datetime('2025-08-19')
        second_frame_change = (datetime > ti) and (datetime < tf)

        corrupted, lake_not_visible, fog=False,False,False
        if first_frame_change or second_frame_change:
            cant_load=True
        else:
            try:
                xnorm, max_lake, diff = get_met(f, datetime)
                corrupted = xnorm > 30
                lake_not_visible = max_lake < 240
                fog = diff < 15
            except:
                cant_load = True

        if (fog or lake_not_visible) or (corrupted or cant_load):
            shutil.move(f,scratch_dir + '/poas_bad/'+fname)
        else:
            shutil.move(f,scratch_dir + '/poas_good/'+fname)          
            
if __name__=='__main__':
    pool = Pool(processes=num_processes)
    pool.map(filter_images, subfiles)