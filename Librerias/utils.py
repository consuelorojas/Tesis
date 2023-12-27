#funciones auxiliares para el programa principal
import numpy as np
import scipy.io
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

def  min_in_subset(serie, index, interval):
    '''
    Find the minimum values in the given series.

    Parameters:
    -----------
    series (pandas.Series):
      The input series.
    index (int):
      The index to search.
    interval (int):
      The interval to search.

    Returns:
    --------
    The minimum value in the given series.
    '''
    min1 = np.argmin(np.abs(serie[index-interval:index]))
    min2 = np.argmin(np.abs(serie[index:index+interval]))

    return min1+index-interval, min2+index
   

def min_in_arrays(array_frame, serie, array, interval=15):
  '''
  Find the minimum values within a specified interval for each element in an array.

  Parameters:
  -----------
  array_frame (list):
    A list of arrays or data frames.
  serie (int):
    The index of the series within each array or data frame.
  array (list):
    A list of elements.
  interval (int, optional):
    The interval size. Defaults to 15.

  Returns:
  --------
  pandas.DataFrame:
    A DataFrame containing the start and end values of the minimum intervals.

  '''
  min1 = []
  min2 = []

  for i in range(len(array)):
    subset = array_frame[i]
    aux1, aux2 = min_in_subset(subset[serie], array[i], interval)
    min1.append(aux1)
    min2.append(aux2)

  minimuns = pd.DataFrame({'start': min1, 'end': min2})
  return minimuns

def get_tau(array, df_general, df_min):
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
  for i in range(len(array)):
    x = array[i]
    y = df_min['end'].iloc[i]
    subset1 = df_general[i]

    num1 =  np.abs(subset1['Gradient Phase'].iloc[x])/2
    subset2 = np.abs(subset1['Gradient Phase'].iloc[x:y])
    num2 = find_nearest(subset2, num1)

    indices2 = subset2[np.abs(subset2 == num2)].index[0]
    indices_tau.append(indices2)
  return indices_tau

