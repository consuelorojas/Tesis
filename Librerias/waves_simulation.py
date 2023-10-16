
import numpy as np
import scipy.signal as signal
import random

#set_seed = random.seed(42)
#print('toi funcionando')

class SimulatedSignal:
    '''
    Creates a noise faraday wave signal with a given frecuency
    ---------------
    Parameters:
    @ freq: frecuency of the wave and its harmonics
    @ cutoff: cutoff frequency of the noise (default 1/8)
    @ t: duration of the signal (default 10.000 seg)
    @ fs : number of points of the signal (default 10.000)
    ---------------
    Returns:
    @ signal: simulated signal
    '''

    def __init__(self, freq, cutoff=1/8, t =  10.0, fs = 10000):
        self.freq = freq
        self.cutoff = cutoff
        self.t = t
        self.fs = fs
        self.random = random.randint(0,1000)

    def faradayWave(self, height = 5*(10**-3), harmonics = 15):
        '''
        Creates a faraday wave signal with a given frecuency
        ---------------
        Parameters:
        @ height: amplitude of the wave (default 5mm)
        @ harmonics: number of harmonics of the wave (default 15)
        ---------------
        Returns:
        @ wave: faraday wave signal
        '''
        time_array = np.linspace(0,self.t,self.fs)

        #1D Faraday wave
        omega = self.freq * 2 * np.pi
        wave = height * np.cos(omega*time_array)

        for i in range (1, harmonics):
            
            wave +=  (height/(i+1)) * np.sin((i+1)*omega*time_array)

        return wave
    
    def noise(self):
        '''
        Creates a noise signal with a given cutoff frequency
        ---------------
        Parameters:
        @ cutoff: cutoff frequency of the noise (default 1/8)
        ---------------
        Returns:
        @ noise: noise signal
        '''
        random.seed(self.random)
        n = np.random.normal(0, 1, self.fs)/100
        b, a = signal.butter(3, self.cutoff, btype='low', analog=False)
        noise = signal.filtfilt(b, a, n)

        return noise
    
    def signal(self):
        '''
        Creates a noisy faraday wave signal with a given frecuency
        ---------------
        Parameters:
        @ freq: frecuency of the wave and its harmonics
        @ cutoff: cutoff frequency of the noise (default 1/8)
        @ t: duration of the signal (default 10.000 seg)
        @ fs : number of points of the signal (default 10.000)
        ---------------
        Returns:
        @ signal: simulated signal
        '''
        return self.faradayWave() + self.noise()




