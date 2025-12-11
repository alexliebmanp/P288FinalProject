
from obspy import Inventory, UTCDateTime, read
from obspy.clients.fdsn import Client
date_range = ["2025-01-22", "2025-03-15"] # change if desired
dpath = '/Users/oxide/Documents/research/orenstein/code/P288FinalProject/data/'

##
## Function definitions
##

def get_inventory(station, dates, channel='*', level='channel', network='OV'):
    '''
    get inventory for a given station
    '''

    client = Client("IRIS")

    starttime = UTCDateTime(f'{dates[0]}')
    endtime = UTCDateTime(f'{dates[1]}')+86400
    inventory = client.get_stations(
        network=network,
        station=station,
        starttime=starttime,
        endtime=endtime,
        channel=channel,
        level=level
    )
    
    return inventory

def get_stream(station, dates, channel, network='OV', kwargs={}):

    client = Client("IRIS")

    starttime = UTCDateTime(f'{dates[0]}')
    endtime = UTCDateTime(f'{dates[1]}')+86400

    # Get a day of waveform data from the data center
    stream = client.get_waveforms(
        network=network,
        station=station,
        location="*",
        channel=channel,
        starttime=starttime,
        endtime=endtime,
        **kwargs
    )
    stream.merge(method=-1)

    return stream

def check_response(station, dates):
    '''
    Given an inventory, check whether there is any corresponding response to any of the channels
    '''

    inventory = get_inventory(station, dates)

    # Check each channel
    for network in inventory:
        for station in network:
            print(f"\nStation: {station.code}")
            for channel in station:
                print(f"\n  Channel: {channel.code} ({channel.location_code})")
                
                if channel.response:
                    print(f"    ✓ Has response data")
                    sens = channel.response.instrument_sensitivity
                    print(f"    Sensitivity: {sens.value} {sens.input_units}/{sens.output_units}")
                    print(f"    Frequency: {sens.frequency} Hz")
                    
                    # Check if it's a full response or just sensitivity
                    if len(channel.response.response_stages) > 1:
                        print(f"    Full response ({len(channel.response.response_stages)} stages)")
                    else:
                        print(f"    Simple response (sensitivity only)")
                else:
                    print(f"    ✗ No response data available")

def get_seismic_data(station, dates, network='OV', datapath=dpath):
    '''
    Get seismic data

    channel: 'HH*' for seismic, 'LF*' for magnetic
    '''

    client = Client("IRIS")

    # Get the instrument response inventory for a single station - don't think I need it in the end
    inventory = Inventory()
    inventory += get_inventory(station, dates)
    inventory.write(f'{datapath}{station}_mag_response.xml', format="STATIONXML")

    # Get a day of waveform data from the data center
    stream = get_stream(station, dates, channel='HH*', kwargs={'attach_response':True})

    # remove the response
    stream.remove_response(output='VEL', pre_filt=(0.01, 0.02, 20, 40), water_level=60)

    # Write each component to a separate miniSEED file
    # E,N,Z refers to three orientations of the seismometers
    for component in "ENZ":
        component_stream = stream.select(component=component)
        component_stream.write(f'{datapath}{network}.{station}_HH{component}.m', format="MSEED")

    return stream

def get_magnetic_data(station, dates, network='OV', datapath=dpath):
    '''
    Get seismic data

    channel: 'HH*' for seismic, 'LF*' for magnetic
    '''

    client = Client("IRIS")

    # Get a day of waveform data from the data center
    stream = get_stream(station, dates, channel='LF*', kwargs={'attach_response':True})

    stream.remove_sensitivity()

    # Write each component to a separate miniSEED file
    # E,N,Z refers to three orientations of the seismometers
    for component in "ENZ":
        component_stream = stream.select(component=component)
        component_stream.write(f'{datapath}{network}.{station}_LF{component}.m', format="MSEED")

    return stream

##
## Pull magnetic data
##
mag_VPRS = get_magnetic_data('VPRS', date_range)
