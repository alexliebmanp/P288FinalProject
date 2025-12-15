import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import imread
import os, glob, shutil
import xarray as xr
from tqdm import tqdm
from scipy.stats import mode
from scipy.ndimage import laplace, sobel, binary_fill_holes, binary_dilation, label
from skimage.segmentation import flood
from scipy.signal import argrelmax, argrelmin
from multiprocessing import Pool

## function of file: create masks over the two lakes from composites

## load images from file directory (composites)
def load_images(fd):
    files = sorted(glob.glob(fd+'/*.jpg'))
    timestr = [' '.join(f.split('/')[-1].split('.')[0].split('_')) for f in files]
    timestr = np.array([pd.to_datetime(t) for t in timestr], dtype='datetime64[m]')

    dat = np.zeros((len(files), 480, 640))
    for i,f in enumerate(files):
        dat[i] = imread(f)
    
    return timestr,dat

def lake_fill(img,center,mask=None,fstep=None):
    img = img/img.max()
    l = laplace(img)
    
    tols = np.arange(0.1,1,0.01)
    
    fs = np.array([flood(img, center, tolerance=tol) for tol in tols])
    As = fs.sum(axis=(1,2))

    avg = (img*fs).sum(axis=(1,2))/As
    davg = np.gradient(avg)

    dfs = np.gradient(np.where(fs,1,0), axis=0)
    dimg = (dfs*img).sum(axis=(1,2))/dfs.sum(axis=(1,2))

    dl = (dfs*l).sum(axis=(1,2))/dfs.sum(axis=(1,2))

    dAs = np.gradient(As)
    fdAs = dAs/As

    # checks if mask encounters any bonudaries
    dl = np.where(np.isnan(dl),-np.inf,dl)
    i = np.argmax(dl)
    
    retvars = [dl, fdAs,dimg]

    ## lake mask can't escape max extent of lake
    if mask is not None:
        outside = (fs*(1-mask)).sum(axis=(1,2))
        oi = np.argmax(np.where(outside>0,1,0))
        
        i = np.minimum(oi,i)
        retvars += [outside]
        
    ## prevents extreme growth
    if fstep is not None:
        ignore = np.where(np.gradient(dimg)>0,0,1)
        A_arr = np.where(np.logical_and(fdAs > fstep,
                                           np.gradient(fdAs) > 0),1,0)


        
        Ai = np.argmax(A_arr)
        if not np.any(A_arr):
            Ai = 1000
        i = np.minimum(Ai,i)

    if i == 0:
        return fs[i]*0, retvars
    return fs[i],retvars

    
scratch_dir = '/pscratch/sd/d/dcarrel/volcano_imgs/'
fft_lodir = scratch_dir + '6hourly_lofft'
times, lo = load_images(fft_lodir)
lo = xr.DataArray(lo, coords={'time': times, 'y': np.arange(480), 'x': np.arange(640)})

## creates composite and mask of max lake extent
loro = lo.rolling({'time':5}, center=True,min_periods=None).max()
lmax = lo.quantile(0.99, dim='time')
lake1 = (340,285)
mask = lake_fill(lmax.values, lake1)[0]
mask = binary_dilation(mask, iterations=10) ## pads by 10 pixels
mask = binary_fill_holes(mask)

## finds location of max pixel value
## near lake
def find_max(img, X,Y,sl):
    img = img[sl].ravel()
    X = X[sl].ravel()
    Y=Y[sl].ravel()
    am = np.argmax(img)
    return (X[am],Y[am])

## empty arrays, stop at 2025/04 since camera shifts and becomes unfocused
loro_gt = loro.where(loro.time < pd.to_datetime('2025-04-20'), drop=True)
l2f = np.zeros_like(loro_gt.values)
l1f = np.zeros_like(loro_gt.values)

## calcs mask for both lakes
def make_mask(t):
    box1 = (slice(335,345), slice(280,300))
    box2 = (slice(280,300), slice(190,215))
    im = loro.sel(time=t).values
    Y,X=np.meshgrid(np.arange(480), np.arange(640))
    lake1= find_max(im, X,Y,box1)
    lake2 = find_max(im, X,Y,box2)
    mask2,*_ = lake_fill(im, mask=mask, center=lake2, fstep=0.3)
    mask1,*_ = lake_fill(im, mask=mask,center=lake1, fstep=0.3)
    return mask1, mask2

num_processes = 32
times_split = np.array_split(loro_gt.time.values, num_processes)

def make_array(ts):
    for t in tqdm(ts):
        tstr = pd.to_datetime(t).strftime('%Y%m%dT%H%M')
        mask1, mask2 = make_mask(t)
        np.save(f'arrs/mask1_{tstr}.npy', mask1)
        np.save(f'arrs/mask2_{tstr}.npy', mask2)

if __name__=='__main__':
    pool = Pool(processes=num_processes)
    pool.map(make_array, times_split)