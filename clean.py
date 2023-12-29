import os
import re
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np
from tqdm import tqdm

def extract_number(filename):
    # Extract the number between "bn" and "_v00" using regular expression
    match = re.search(r'bn(\d+)_v', filename)
    if match:
        return int(match.group(1))
    else:
        return None  # Return None for filenames that don't match the pattern

def lowpass_filter(data, cutoff_frequency_radians):
    nyquist = 0.5 * np.pi  # Nyquist frequency for radians
    normal_cutoff = cutoff_frequency_radians / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Define the source directory
source_directory = r'/home/itay_hadas/bursts'

# Get the list of filenames in the source directory
filenames = [filename for filename in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, filename))]

# Remove '.' and '..' from the list (current and parent directory)
filenames = [filename for filename in filenames if filename not in {'.', '..'}]

# Extract numbers between "bn" and "_v" using the extract_number function
numbers = [extract_number(filename) for filename in filenames]
numbers = [number for number in numbers if number is not None]

# Get unique numbers
unique_numbers = np.unique(numbers)

# Initialize a dictionary to store filenames for each unique number
file_lists = {number: [] for number in unique_numbers}

# Populate file_lists with filenames for each unique number
for file in filenames:
    number = extract_number(file)
    file_lists[number].append(file)

# iterate over all files 
for i in tqdm(range(1, len(unique_numbers))):
    try:
        # extract bcat and time information from bcat
        for j in range(len(filenames) // len(unique_numbers)):
            if 'bcat' in file_lists[unique_numbers[i]][j]:
                curr_bcat = file_lists[unique_numbers[i]][j]
                file_lists[unique_numbers[i]].remove(curr_bcat)
                break

        m_data = fits.getheader(os.path.join(source_directory, curr_bcat))
        m_data_2 = fits.getheader(os.path.join(source_directory, file_lists[unique_numbers[i]][0]))

        trig_start_time = m_data_2['TSTART']
        trig_time = m_data_2['TRIGTIME'] - trig_start_time
        trig_end_time = m_data_2['TSTOP'] - trig_start_time
        trig_start_time = 0
        resolution = 10
        bins = round(resolution * (trig_end_time - trig_start_time))

        t90 = m_data['T90']
        terr = m_data['T90_ERR']
        burst_start = trig_time + m_data['T90START']

        max_val = 0
        max_times = 0
        x_axis = np.linspace(trig_start_time, trig_end_time, bins)

        # take burst from sensor with highest value
        for j in range(len(file_lists[unique_numbers[i]])):
            curve = fits.getdata(os.path.join(source_directory, file_lists[unique_numbers[i]][j]), ext=2)
            try:
                times = curve['TIME']
            except:
                break
            values, _ = np.histogram(times, bins)

            if np.max(values) > max_val:
                max_times = values
                max_val = np.max(values)
        
        # pass through lpf, and filter out cases where start is higher then end
        max_times = lowpass_filter(max_times, 0.1)
        start_std_value = np.mean(max_times[4:14]) + np.std(max_times[4:14])
        end_std_value = np.mean(max_times[-15:-5]) + np.std(max_times[-15:-5])
        ratio = np.mean(max_times[4:14]) / np.mean(max_times[-15:-5])

        if (ratio < 1.5 and 1 / ratio < 1.5):
            # elongate burst from the end until the std is like in the end of the full uncropped burst 
            new_std = end_std_value
            max_time = np.argmax(max_times)
            curr_time_end = round(max([trig_time * resolution, resolution * (burst_start + t90), max_time]))
            curr_time_start = round(burst_start*resolution)
            while new_std >= end_std_value:
                if curr_time_end + 5 < round(trig_end_time * resolution):
                    new_std = np.mean(max_times[curr_time_end:curr_time_end + 5])
                    curr_time_end = curr_time_end + 5
                else:
                    curr_time_end = round(trig_end_time * resolution) - 5
                    break

            if curr_time_end < round(trig_end_time * resolution) - 15:
                curr_time_end = curr_time_end + 10

            curr_time_start = curr_time_start - 10

            indices = list(range(curr_time_start, curr_time_end + 1))
            max_times = np.array(max_times)
            max_times = max_times[indices]

            # save cleaned burst
            name = os.path.join('./clean_bursts_filtered', f"{unique_numbers[i]}")
            np.save(name, max_times)
        else:
            print(f'failed ratio cutoff: {unique_numbers[i]}')

    except:
        print(f'failed for techincal reasons: {unique_numbers[i]}')

        