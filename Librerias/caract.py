import pandas as pd
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, hilbert

import sys
sys.path.append('.../data')
sys.path.append('.../Librerias')

import Librerias.utils as utils


#class
class CaractDefect:
    '''A class that calculates the characteristics of a defect in the signal
    '''


    def __init__(self, df, fs=1000):
        '''
        Initializes the CaractDefect object.

        Parameters:
        -----------
        df : pandas DataFrame
            The DataFrame with the original signal, signal minus mean, filtered signal, and Hilbert transform.
        fs : int
            The sampling frequency of the signal.
        '''
        self.df = df
        self.fs = fs
    
    def get_hilbert(self, cutoff = 0.1, order = 4):
        '''
        This method applies the Hilbert transform to the signal data stored in the DataFrame.
        It calculates the amplitude, instantaneous phase, and gradient phase of the signal.
        The results are returned as a new DataFrame.

        Parameters:
        ------------------
        cutoff (float): 
            The cutoff frequency for the lowpass Butterworth filter. Default is 0.1.
        order (int): 
            The order of the lowpass Butterworth filter. Default is 4.

        Returns:
        ------------------
        pandas.DataFrame: 
            A DataFrame containing the Hilbert Transform, Amplitude, Instantaneous Phase,
        and Gradient Phase of the signal.
        '''
        df = self.df
        amplitude = np.abs(df['Hilbert Transform'])
        int_phase = np.unwrap(np.angle(df['Hilbert Transform']))
        diff_phase  = np.diff(int_phase)
        mean_phase = np.mean(diff_phase)
        diff_phase = np.insert(diff_phase, 0, 0)


        b, a = butter(order, cutoff, fs=self.fs, btype='lowpass')
        grad_phase = filtfilt(b, a, diff_phase-mean_phase)

        hilbert_pd = pd.DataFrame({'Hilbert Transform': df['Hilbert Transform'],
                                   'Amplitude': amplitude,
                                   'Instantaneous Phase': int_phase,
                                   'Gradient Phase': grad_phase})
        
        return hilbert_pd
    
    def get_peaks(self, cutoff = 0.1, order = 4):
        '''
        This method calculates the peaks in the signal data stored in the DataFrame.
        It returns the peaks (index) as a new DataFrame.

        Returns:
        ------------------
        pandas.DataFrame: 
            A DataFrame containing the peaks of the signal.
        '''
        df = self.df
        hilbert =  self.get_hilbert(cutoff, order)
        peaks, _ = scipy.signal.find_peaks(hilbert['Gradient Phase'], distance=10)
        peaks_pd = pd.DataFrame({'Peaks': peaks})
        return peaks_pd
    
    def get_minAmp(self, mult = 30):
        '''
        This method calculates the minimum amplitude in the signal data stored in the DataFrame.
        It returns the minimum amplitude (index) as a new DataFrame.

        Parameters:
        ------------------
        mult (float): 
            The multiplier for the minimum amplitude. Default is 30.

        Returns:
        ------------------
        pandas.DataFrame: 
            A DataFrame containing the minimum amplitude of the signal.
        '''
        frame = self.get_hilbert()
        min_amp = frame['Amplitude'].min()
        indices = frame[frame['Amplitude'] <= min_amp*mult].index
        indices = pd.DataFrame({'Min Amp': indices})
        return indices

    def get_defectos(self, cutoff=0.1, order=4, mult=30):
        """
        Returns the intersection indices and concatenated dataframes of peaks, minimum amplitude, and hilbert values.

        Parameters:
        -----------
        cutoff (float): 
            The cutoff value for peak detection. Default is 0.1.
        order (int): 
            The order of the peak detection filter. Default is 4.
        mult (int): 
            The multiplier for minimum amplitude detection. Default is 30.

        Returns:
        --------
        intersection (numpy.ndarray):
            The intersection indices of peaks, minimum amplitude, and hilbert values.
        defectos_pd (list):
            List of concatenated dataframes containing x and y values for each intersection index.
        """
        peaks = self.get_peaks(cutoff, order)
        min_amp = self.get_minAmp(mult)
        hilbert = self.get_hilbert(cutoff, order)

        intersection = np.intersect1d(peaks, min_amp, assume_unique=False, return_indices=False)

        defectos_pd = []
        for i in intersection:
            x = self.df.iloc[i - 500 : i + 500]
            y = hilbert.iloc[i - 500 : i + 500]

            defectos_pd.append(pd.concat([x, y], axis=1))

        return intersection, defectos_pd
    
