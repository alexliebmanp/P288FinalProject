import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import imread
import os, glob
import xarray as xr
from tqdm import tqdm
import shutil
from scipy import fft
from PIL import Image
from multiprocessing import Pool

scratch_dir = '/pscratch/sd/d/dcarrel/'

## load in "good" files
im_files = sorted(glob.glob(scratch_dir+'poas_good/*.jpg'))
times = [pd.to_datetime(f.split('.')[0].split('/')[-1]) for f in im_files]
times = np.array(times, dtype='datetime64[s]')
times_rounded = np.array(times, dtype='datetime64[h]')

freq = pd.Timedelta('6h') ## composite frequency/width
times_ideal = pd.date_range(times.min(), times.max(), freq=freq)
times_ideal=np.array(times_ideal, dtype='datetime64[h]')
times_to_compute = np.intersect1d(times_rounded, times_ideal)

files = xr.DataArray(im_files, coords={'time':times})

n_processes=16
times_per_process = np.array_split(times_to_compute, n_processes)

## calculate high and low frequency parts of image timeseries
def calc_save_statistics(times):
    for time in tqdm(times):

        time = pd.to_datetime(time)
        start_time = time - freq/2
        end_time = time + freq/2
        fs = files.loc[{'time': slice(start_time, end_time)}]
        ftimes = fs.time.values
        fs = fs.values

        ## check if have enough data points
        pct = len(fs) / int(freq/pd.Timedelta('1m'))
        if pct < 0.4:
            continue
            
        arr = np.zeros((len(fs), 480, 640))
        for i,f in enumerate(fs):
            arr[i] = imread(f)

        transformed = np.abs(fft.fft(arr, axis=0))**2
        fftfreq = np.abs(fft.fftfreq(transformed.shape[0]))
        lf = fftfreq < 0.1
        hf = fftfreq > 0.4
        
        lowfreq = transformed[lf].mean(axis=0)
        lowfreq /= lowfreq.max()
        lowfreq = (255*lowfreq).astype(np.uint8)

        hifreq = transformed[hf].mean(axis=0)
        hifreq /= hifreq.max()
        hifreq = (255*hifreq).astype(np.uint8)
       # p95 = np.percentile(arr, 95, axis=0).astype(np.uint8)
       # p80 = np.percentile(arr, 80, axis=0).astype(np.uint8)
       # p50 = np.median(arr, axis=0).astype(np.uint8)

        tstr = time.strftime('%Y%m%d_%H%M')
        ## now save
        Image.fromarray(lowfreq).save(scratch_dir+f'/volcano_imgs/6hourly_lofft/{tstr}.jpg')
        Image.fromarray(hifreq).save(scratch_dir+f'/volcano_imgs/6hourly_hifft/{tstr}.jpg')
        #Image.fromarray(p95).save(scratch_dir+f'/volcano_imgs/6hourly_95/{tstr}.jpg')
        #Image.fromarray(p80).save(scratch_dir+f'/volcano_imgs/6hourly_80/{tstr}.jpg')
        #Image.fromarray(p50).save(scratch_dir+f'/volcano_imgs/6hourly_50/{tstr}.jpg')
        #Image.fromarray(median).save(scratch_dir+f'/poas_hourly/median/{tstr}.jpg')
        #except:
        #    continue
    

if __name__=='__main__':
    pool = Pool(processes=n_processes)
    pool.map(calc_save_statistics, times_per_process)
