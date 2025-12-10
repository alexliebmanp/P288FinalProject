# %%

""" Created on November 13, 2025 // @author: Sarah Shi """

import os
import time
import requests

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tqdm import tqdm
from obspy import Inventory, UTCDateTime
from obspy.clients.fdsn import Client

from PIL import Image
from io import BytesIO

# %% Soil CO2 data 

# Combining 2024/2025 data
# co2_2024 = pd.read_csv('Soil_CO2/poas_vppc_2024_processed.csv',
#                        skiprows=3, header=None)
# co2_2025 = pd.read_csv('Soil_CO2/poas_vppc_2025_processed.csv', 
#                        skiprows=3, header=None) 
# co2 = pd.concat([co2_2024, co2_2025])
# co2.columns = ['Date', 'CO2_ppm']
# co2.to_csv('Soil_CO2/poas_vppc_co2.csv')

co2 = pd.read_csv('Soil_CO2/poas_vppc_co2.csv', index_col=0)
co2['Date'] = pd.to_datetime(co2['Date'])
display(co2)

plt.figure(figsize=(8, 6))
plt.plot(co2.Date, co2.CO2_ppm)
plt.xlabel('Date')
plt.ylabel('CO2 (ppm)')
plt.title('Soil CO2 at Poas Volcano')
plt.tight_layout()

# %% Query for IR images

response = requests.get(
    "https://avert-legacy.ldeo.columbia.edu/api/imagery/infrared/query",
    params={
        "site": "VPMI",
        "vnum": 345040,
        "search_from": "2023-11-01T00:00",
        "search_to": "2025-11-01T00:00",
    }
)
images = response.json()

# %% Subsample imagery data

# Print the number of images in this time range
# These images are taken once every minute, so there is an abundance of imagery data

print(len(images))

# Downsample, start by keeping every 2nd image so 1 image / 2 minutes
step = 2
images_ds = images[::step]

print(len(images_ds))

# %% Look through hourly data and save

# Base URL host for image download
base = "https://avert-legacy.ldeo.columbia.edu"
out_dir = "VPMI_hourly"
out_dir_bad = "VPMI_hourly_bad"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir_bad, exist_ok=True)

stdev_min = 15 # tweak this threshold

# Images are stored with a string that is described as follows: 
# 345040.VPMI.YEAR.DAY_OF_YEAR(JULIAN_DAY).HHMMSS.Index
# Julian Day 291 = 2025-10-18 (Y-M-D)
# HHMMSS = Hour minute second 

for img in tqdm(images_ds, desc="Downloading hourly images"):
    # Fix path: API gives /data/vulcand/archive/..., real files are /archive/...
    rel = img["url"].replace("/data/vulcand", "")
    url = base + rel
    filename = os.path.join(out_dir, img["image_id"] + ".jpg")
    filename_bad = os.path.join(out_dir_bad, img["image_id"] + ".jpg")

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print("Failed:", r.status_code, url)
            continue

        # Load image from bytes and compute std-dev
        im = Image.open(BytesIO(r.content)).convert("L")
        arr = np.array(im, dtype=np.float32)
        stdev = float(arr.std())
        # check edge saturation
        left_edge = arr[:, :20].mean()
        right_edge = arr[:, -20:].mean()
        edge = np.mean([left_edge, right_edge])
        center = arr[:, arr.shape[1]//3 : 2*arr.shape[1]//3].mean()

    except Exception as e:
        print("Error reading image:", url, e)
        continue  # skip this one

    # Commented out (NOT SAVING) after some preliminary testing.
    # Started by testing how we could do some simple filters to remove low quality images. 
    # These are grey across the image, or have blown-out edges.
    # Can test saving to confirm that this filtering is working.     
    # Filter out low-variance / junk frames
    if stdev < stdev_min:
        # optional: print or log if you want to see what's being skipped
        # print(f"Skipping {img['image_id']} (std = {stdev:.2f})")
        # with open(filename_bad, "wb") as f:
        #     f.write(r.content)
        continue

    if edge/center > 5: 
        # print(f"Skipping {img['image_id']} (edge/center = {edge/center:.2f})")
        # with open(filename_bad, "wb") as f:
        #     f.write(r.content)
        continue

    # Save only good images
    with open(filename, "wb") as f:
        f.write(r.content)

# %% Seismic data

# Trillium Compact (120s post-hole) seismometers manufactured by Nanometrics

client = Client("IRIS")
stat = "VPCC"

# Get the instrument response inventory for a single station
inventory = Inventory()
inventory += client.get_stations(
    network="OV", # CHANGE THIS TO THE NETWORK YOU WANT TO DOWNLOAD FROM
    station=stat, # CHANGE THIS TO THE STATION YOU WANT TO DOWNLOAD FROM
    starttime=UTCDateTime("2023-11-01"), # CHANGE THIS TO THE START TIME YOU WANT TO DOWNLOAD FROM
    endtime=UTCDateTime("2025-11-01"),
    level="response",
)
inventory.write("VPCC_mag_response.xml", format="STATIONXML")

# Get a day of waveform data from the data center
start_time = time.time()
stream = client.get_waveforms(
    network="OV", # CHANGE THIS TO THE NETWORK YOU WANT TO DOWNLOAD FROM
    station=stat, # CHANGE THIS TO THE STATION YOU WANT TO DOWNLOAD FROM
    location="*",
    channel="HH*,BH*,EH*,HN*",
    starttime=UTCDateTime("2024-11-01"), # CHANGE THIS TO THE START TIME YOU WANT TO DOWNLOAD FROM
    endtime=UTCDateTime("2024-11-02"), # CHANGE THIS TO THE END TIME YOU WANT TO DOWNLOAD FROM
)
stream.merge(method=-1)
time_elapsed = time.time() - start_time
print('Elapsed Time: ', time_elapsed)

# Write each component to a separate miniSEED file
for component in "ENZ":
    component_stream = stream.select(component=component)
    component_stream.write(f"OV.{stat}_{component}.m", format="MSEED")

# %% 

print(stream)  # sanity check
stream.plot(equal_scale=False)


# %%
