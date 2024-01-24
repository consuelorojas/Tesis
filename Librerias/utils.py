#funciones auxiliares para el programa principal
import numpy as np
import pandas as pd

def find_nearest(array, value):
    '''
    Find the element in the given array that is closest to the specified value.

    Parameters:
    -----------
    array (array-like):
      The input array.
    value:
      The value to compare against.

    Returns:
    --------
    The element in the array that is closest to the specified value.
    '''

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_indices_min(serie, indice, intervalo):
  """
  Returns the indices of the minimum values in a given series within a specified interval around a given index.

  Parameters:
  -----------
    serie (array-like):
      The series of values.
    indice (int):
      The index around which to search for minimum values.
    intervalo (int):
      The interval size around the index to consider.

  Returns:
  ------------
    tuple:
      A tuple containing the indices of the minimum values found. The first element is the index of the minimum value
    within the interval before the given index, and the second element is the index of the minimum value within the interval
    after the given index.
  """
  min1 = np.argmin(np.abs(serie[indice-intervalo: indice]))
  min2 = np.argmin(np.abs(serie[indice: indice+intervalo]))
  return (min1+indice-intervalo), (min2+indice)

   
def get_minimuns(serie, x, interval):
  """
  Returns the start and end indices of the minimum values in the given series within the specified interval.

  Parameters:
  -----------
  serie (pandas.Series):
    The series to search for minimum values.
  x (list):
    The list of values to search for minimum values within the interval.
  interval (int):
    The interval to search for minimum values.

  Returns:
  --------
  pandas.DataFrame:
    A DataFrame containing the start and end indices of the minimum values.
  """
  min1 = []
  min2 = []
  for elem in x:
    aux1, aux2 = get_indices_min(serie, elem, interval)
    min1.append(aux1)
    min2.append(aux2)

  minimuns = pd.DataFrame({'start': min1, 'end': min2})
  return minimuns

def get_indices_tau(array, df_general, df_min):
  """
  Get the indices to get tau values.

  Parameters:
  -----------
  array (list): 
    List of indices.
  df_general (DataFrame):
    DataFrame containing 'Gradient Phase' column.
  df_min (DataFrame):
    DataFrame containing 'end' column.

  Returns:
  --------
  list: 
    List of mid interval indices.
  """
  indices_tau = []
  serie = df_general['Gradient Phase']

  for i in range(len(array)):

    x = array[i]
    y = df_min['end'].iloc[i]

    num1 =  np.abs(serie[x])/2
    subset2 = np.abs(serie[x:y])
    num2 = find_nearest(subset2, num1)

    indices2 = subset2[np.abs(subset2 == num2)].index[0]
    indices_tau.append(indices2)
  return indices_tau


def create_windows(df, size=4000, overlap=500):
  """
  Creates overlapping windows of a specified size from a given DataFrame.
  
  Parameters:
  --------------
    df (DataFrame):
      The input DataFrame.
    size (int):
      The size of each window. Default is 4000.
    overlap (int):
      The overlap between consecutive windows. Default is 500.
  
  Returns:
  ---------------
    list:
      A list of windows, where each window is a subset of the input DataFrame.
  """
  num_windows = (len(df) - size) // (size - overlap) + 1
  windows = []
  for i in range(num_windows):
    start = i * (size - overlap)
    end = start + size
    window = df.iloc[start:end]
    windows.append(window)
  return windows

def find_windows(start_end_defectos, windows):
  """
  Finds the windows that contain the given start and end points of defects.

  Parameters:
  --------------
  start_end_defectos (list):
    A list of tuples representing the start and end points of defects.
  windows (list):
    A list of windows.

  Returns:
  ---------------
  list:
    A list of indices of the windows that contain the defects.
  """
  result = []
  for elem in start_end_defectos:
    for i, window in enumerate(windows):
      if elem[0] >= window.index[0] and elem[1] <= window.index[-1]:
        result.append(i)
        break
  return result


def create_sequences(data, window, horizon, drop_index):
  """
  Create sequences of input and output data for time series forecasting.

  Parameters:
  -----------
  data (numpy.ndarray):
    The input time series data.
  window (int):
    The length of the input sequence.
  horizon (int):
    The length of the output sequence.

  Returns:
  -----------
  numpy.ndarray:
    The input sequences.
  numpy.ndarray:
    The output sequences.
  """
  data =  np.delete(data, [drop_index], axis=1)
  xs, ys = [], []
  for i in range(len(data) - window - horizon + 1):
    x = data[i:i+window]
    y = data[i+window: i+window+horizon]
    xs.append(x)
    ys.append(y)

  return np.array(xs), np.array(ys)