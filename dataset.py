'''
Filipe Chagas
12-Feb-2021
'''

import numpy as np
from matplotlib import pyplot as plt
import wavio
from typing import *

def load_wave(filename: str) -> np.ndarray:
    '''
    Load data from the wav file.
    Params:
        filename (str): Name of the wav file.
    
    Returns (nd.ndarray): Array with normalized audio samples 
    '''
    sample_rate, sample_width, data = wavio.readwav(filename)
    assert sample_rate == 44100 #Audio must have 44.1KHz of sample rate

    data = data.transpose()[0] #Change shape (length,1) to (length,)
    data = data / (2**(sample_width*8)) #Normalize values
    return data

def split_wave(data: np.ndarray, samples_per_slice: int) -> List[np.ndarray]:
    '''
    Split wave data to slices.
    Params:
        data (np.ndarray): NumPy array containing audio signal.
        samples_per_slice (int): Quantity of audio samples in each slice.
    
    Returns (List[np.ndarray]): List of slices.
    '''
    assert len(data.shape) == 1 #data shape must be (length,)

    n_samples = data.shape[0] #Quantity of samples in data
    n_slices = n_samples // samples_per_slice #Quantity of slices 

    l = [] #List where the slices will be inserted
    for i in range(n_slices):
        slc = data[i*samples_per_slice : (i+1)*samples_per_slice] #Slice
        l.append(slc)
    
    return l

def load_dataset(x_filename: str, y_filename: str, samples_per_input: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load wave files as a dataset for keras.
    Params:
        x_filename (str): Name of the wave file with input data.
        y_filename (str): Name of the wave file with target data.
        samples_per_slice (int): Quantity of samples in each NN input.
    '''
    x_data = load_wave(x_filename)
    y_data = load_wave(y_filename)

    x_slices = split_wave(x_data, samples_per_input)
    y_slices = split_wave(y_data, samples_per_input)
    
    x_dataset = np.vstack(x_slices)
    y_dataset = np.vstack(y_slices)
    
    return (x_dataset, y_dataset)