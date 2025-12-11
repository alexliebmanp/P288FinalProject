# %% 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from obspy import Inventory, UTCDateTime, read
from obspy.clients.fdsn import Client
dpath = 'data/'
processed_dpath = 'processed_data/'

# %% 

##
## Function definitions
##

def deduplicate_datetime_index(df, strategy='first'):
    """
    Remove duplicate timestamps from a DataFrame, handling both numeric and non-numeric data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex that may contain duplicates
    strategy : str or dict
        How to handle duplicates:
        - 'first': Keep first occurrence (default)
        - 'last': Keep last occurrence
        - 'mean': Average numeric columns, first for non-numeric
        - 'median': Median for numeric, first for non-numeric
        - dict: Column-specific strategies, e.g., {'col1': 'mean', 'col2': 'first'}
    
    Returns
    -------
    pd.DataFrame
        DataFrame with unique timestamps
    """
    if not df.index.has_duplicates:
        return df.copy()
    
    if strategy == 'first':
        return df[~df.index.duplicated(keep='first')]
    
    elif strategy == 'last':
        return df[~df.index.duplicated(keep='last')]
    
    elif strategy in ['mean', 'median']:
        # Group by index and apply appropriate aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        result_parts = []
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if strategy == 'mean':
                numeric_agg = df[numeric_cols].groupby(level=0).mean()
            else:  # median
                numeric_agg = df[numeric_cols].groupby(level=0).median()
            result_parts.append(numeric_agg)
        
        # Handle non-numeric columns (take first)
        if len(non_numeric_cols) > 0:
            non_numeric_agg = df[non_numeric_cols].groupby(level=0).first()
            result_parts.append(non_numeric_agg)
        
        # Combine and restore original column order
        result = pd.concat(result_parts, axis=1)
        return result[df.columns]
    
    elif isinstance(strategy, dict):
        # Column-specific strategies
        result_parts = []
        
        for col in df.columns:
            col_strategy = strategy.get(col, 'first')
            
            if col_strategy == 'first':
                agg_col = df[[col]].groupby(level=0).first()
            elif col_strategy == 'last':
                agg_col = df[[col]].groupby(level=0).last()
            elif col_strategy == 'mean':
                agg_col = df[[col]].groupby(level=0).mean()
            elif col_strategy == 'median':
                agg_col = df[[col]].groupby(level=0).median()
            elif col_strategy == 'min':
                agg_col = df[[col]].groupby(level=0).min()
            elif col_strategy == 'max':
                agg_col = df[[col]].groupby(level=0).max()
            elif col_strategy == 'sum':
                agg_col = df[[col]].groupby(level=0).sum()
            else:
                raise ValueError(f"Unknown strategy '{col_strategy}' for column '{col}'")
            
            result_parts.append(agg_col)
        
        return pd.concat(result_parts, axis=1)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def parse_dates(date_string):
    # Try different format patterns
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",       # Format 1
        "%Y-%m-%d_%H%M%S",            # Format 2
        "%Y-%m-%d %H:%M:%S"           # Format 3
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

def set_datetime(df, time_index='Datetime'):
    """
    sets uniform datetime format for pandas dataframes and sets the datetime as index.
    """
    df[time_index] = pd.to_datetime(df[time_index],utc=True)
    df = df.set_index(time_index)
    # Handle duplicates
    df = deduplicate_datetime_index(df)
    return df

def interpolate_to_index(df, target_index, method='linear'):
    """
    Interpolate dataframe to a new DatetimeIndex.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    target_index : pd.DatetimeIndex
        Target index to interpolate to
    method : str
        Interpolation method: 'linear', 'time', 'nearest', 'cubic', etc.
    
    Returns
    -------
    pd.DataFrame
        Interpolated dataframe
    """

    # Combine original and target indices, then sort
    combined_index = df.index.union(target_index).sort_values()
    
    # Reindex to combined index (fills with NaN)
    df_reindexed = df.reindex(combined_index)
    
    # Interpolate (method='time' is good for irregular time series)
    df_interpolated = df_reindexed.interpolate(method=method)
    
    # Select only the target index
    return df_interpolated.reindex(target_index)

def combine_and_interpolate(dataframes, target_times):
    """
    Interpolates data in dataframes onto target_times. Each dataframe must have 'Datetime' as index.
    """
    
    interpolated_dfs = []
    for df in dataframes:
        
        # interpolate and append - can change interpolate_to_index function for different desired interpolation
        interpolated_dfs.append(interpolate_to_index(df, target_times))
    
    interpolated = pd.concat(interpolated_dfs, axis=1)

    # remove non-numerical columns
    interpolated = interpolated.select_dtypes(include=[np.number])

    return interpolated

def resample_acoustic(data, target_index, dt='1min'):
    """
    resample acoustic data into magnitude vs time onto a specific target grid
    """
    
    magnitude = pd.Series(0.0, index=target_index)
    bin_duration_seconds = pd.Timedelta(dt).total_seconds()
    for _, event in data.iterrows():
        t1 = event['Acoustic_t1']
        t2 = event['Acoustic_t2']
        
        value = event['Magnitude'] # can change this line to redefine "eruption activity"

        affected_bins = target_index[(target_index <= t2) & (target_index + pd.Timedelta(dt) >= t1)]
        
        for bin_start in affected_bins:
            bin_end = bin_start + pd.Timedelta(dt)
            overlap_start = max(t1, bin_start)
            overlap_end = min(t2, bin_end)
            overlap_seconds = (overlap_end - overlap_start).total_seconds()
            
            if overlap_seconds > 0:
                fraction = overlap_seconds / bin_duration_seconds # smoothing of the data a bit
                magnitude[bin_start] += value * fraction

    magnitude = magnitude.to_frame(name='Eruption_Activity')
    
    return magnitude

##
## Load data - to add more data to the pipeline, add it here!
##

date_range = pd.to_datetime(["2024-01-31 0:0:0.0", "2025-12-01 00:00:00.0"], utc=True)

# load POAS acoustic data
acoustic = pd.read_csv(dpath+'Poas_discrete_2023-01-01_2025-12-12.csv', header=1, names=['Event ID', 'Label', 'Station', 'Sensor Type','Acoustic_t1', 'Acoustic_t2', 'Energy' ,'Duration', 'Magnitude'])
acoustic['Acoustic_t1'] = pd.to_datetime(acoustic['Acoustic_t1'],utc=True)
acoustic['Acoustic_t2'] = pd.to_datetime(acoustic['Acoustic_t2'],utc=True)
acoustic['Datetime'] = pd.to_datetime((acoustic['Acoustic_t1'].astype('int64') + acoustic['Acoustic_t2'].astype('int64')) / 2) # mean time
acoustic = set_datetime(acoustic)
acoustic.name = 'acoustic'

# load seismic data from Sarah
seismic_data = []
for station in ['VPCC', 'VPPC', 'VPNC', 'VPRS']: #
    df = pd.read_csv(dpath+f'{station}_RSAM_600s_merged.csv', header=2, names=['Datetime', f'{station}_RSAM', 'Station', 'Channel', 'Year', 'Month'])
    df = set_datetime(df)
    df = df.drop(columns=['Year', 'Month'])
    df.name = f'{station}_seismic'
    seismic_data.append(df)

# load magnetic data
magnetic_data = []
for station in ['VPRS']:
    traceE = read(dpath+f'OV.{station}_LFE.m')[0]
    traceN = read(dpath+f'OV.{station}_LFN.m')[0]
    traceZ = read(dpath+f'OV.{station}_LFZ.m')[0]
    timeseriesE = np.array(traceE)
    timeseriesN = np.array(traceN)
    timeseriesZ = np.array(traceZ)
    times = np.array([t.datetime for t in traceE.times(type='utcdatetime')])
    column_names = ['Datetime', f'{station} Field E', f'{station} Field N', f'{station} Field Z']
    df = pd.DataFrame(zip(times, timeseriesE, timeseriesN, timeseriesZ), columns=column_names)
    df = set_datetime(df)
    df.name = f'{station}_magnetic'
    magnetic_data.append(df)

# load soil CO2 data
soilCO2 = pd.read_csv(dpath+'CO2.csv', header=1, names=['Datetime', 'CO2_ppm'])
soilCO2 = set_datetime(soilCO2)
soilCO2.name = 'soilCO2'

# load weather
df1 = pd.read_csv(dpath+'VPMI_weather_2024.csv', delimiter=',', header=0)
df2 = pd.read_csv(dpath+'VPMI_weather_2025.csv', delimiter=',', header=0)
weather = pd.concat([df1, df2])
weather['Datetime'] = [parse_dates(dt) for dt in weather['Datetime']]
weather = set_datetime(weather)
weather.name = 'weather'

##
## combine data and save dataset
##

# define times over which to interpolate (1 minute increments between start and end date)
target_times = pd.date_range(date_range[0], date_range[1], freq='min')

# interpolate acoustic data
acoustic_interpolated = resample_acoustic(acoustic, target_times)

# combine and interpolate
dataframes = [*seismic_data, *magnetic_data, soilCO2, weather, acoustic_interpolated] # all dataframes to combine
interpolated = combine_and_interpolate(dataframes, target_times)
interpolated.name = 'dataInterpolated'

# save combined interpolated data as CSV
interpolated.to_csv(processed_dpath+f'{interpolated.name}.csv')


# %%

plt.figure(figsize=(12,8))
plt.plot(interpolated.index, interpolated.CO2_ppm, label='CO2')
plt.semilogy(interpolated.index, interpolated["VPCC_RSAM"], label='VPCC_RSAM')


# %% 