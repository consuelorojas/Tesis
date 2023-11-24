import pandas as pd
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, hilbert

import sys
sys.path.append('.../data')

class MatFileToDataFrame:
    '''
    A class that converts a MATLAB .mat file to a pandas DataFrame with additional signal processing steps.

    Attributes:
    -----------
    file_path : str
        The path to the .mat file to be converted.
    variable_name : str
        The name of the variable in the .mat file to be converted.
    cutoff : float or tuple
        The cutoff frequency or frequencies for the bandpass filter.

    Methods:
    --------
    get_dataframe(cutoff)
        Returns a pandas DataFrame with the original signal, signal minus mean, filtered signal, and Hilbert transform.
    butter_bandpass_filter(signal, cutoff, fs=1000, order=1)
        Applies a Butterworth bandpass filter to the input signal.
    hilbert_transform(signal)
        Applies a Hilbert transform to the input signal to obtain the amplitude envelope.
    '''
    def __init__(self, file_path, file_name):
        '''
        Initializes the MatFileToDataFrame object.

        Parameters:
        -----------
        file_path : str
            The path to the .mat file to be converted.
        variable_name : str
            The name of the variable in the .mat file to be converted.
        cutoff : float or tuple
            The cutoff frequency or frequencies for the bandpass filter.
        '''
        self.file_path = file_path
        self.file_name = file_name
    
    def get_dataframe(self, cutoff):
        '''
        Returns a pandas DataFrame with the original signal, signal minus mean, filtered signal, and Hilbert transform.

        Parameters:
        -----------
        cutoff : float or tuple
            The cutoff frequency or frequencies for the bandpass filter.

        Returns:
        --------
        df : pandas DataFrame
            A DataFrame with the original signal, signal minus mean, filtered signal, and Hilbert transform.
        '''
        mat_data = scipy.io.loadmat(self.file_path+self.file_name)
        signal = pd.DataFrame(mat_data['data'])
        signal_mean = signal.mean(axis=0)
        signal_filtered = self.butter_bandpass_filter(signal, cutoff)
        signal_hilbert = self.hilbert_transform(signal_filtered)
        df = pd.concat([signal, signal-signal_mean, signal_filtered, signal_hilbert], axis=1)
        df.columns = ['Original Signal', 'Signal - Mean', 'Filtered Signal', 'Hilbert Transform']
        return df
    
    def butter_bandpass_filter(self, signal, cutoff, order=4):
        '''
        Applies a Butterworth bandpass filter to the input signal.

        Parameters:
        -----------
        signal : pandas DataFrame
            The input signal to be filtered.
        cutoff : float or tuple
            The cutoff frequency or frequencies for the bandpass filter.
        fs : int, optional
            The sampling frequency of the input signal. Default is 1000 Hz.
        order : int, optional
            The order of the Butterworth filter. Default is 1.

        Returns:
        --------
        signal_filtered : pandas DataFrame
            The filtered signal.
        '''
        #nyq = 0.5 * fs
        normal_cutoff = cutoff #/ nyq
    
        if len(cutoff) > 1:
            #pasa baja
            a, b = butter(order, normal_cutoff[1], btype='low', analog=False)
            sig1 =  filtfilt(a, b, signal, axis=0)
            #pasa alta
            c, d = butter(order, normal_cutoff[0], btype='high', analog=False)
            signal_filtered = filtfilt(c, d, sig1, axis=0)

        elif len(cutoff) == 1:
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            signal_filtered = filtfilt(b, a, signal, axis=0)
        
        return pd.DataFrame(signal_filtered)
    
    def hilbert_transform(self, signal):
        '''
        Applies a Hilbert transform to the input signal to obtain the amplitude envelope.

        Parameters:
        -----------
        signal : pandas DataFrame
            The input signal to be transformed.

        Returns:
        --------
        analytic_signal : pandas DataFrame
            The Hilbert Transform of the input signal.
        '''
        analytic_signal = hilbert(signal, axis=0)
        analytic_signal = pd.DataFrame(analytic_signal)
        return analytic_signal
