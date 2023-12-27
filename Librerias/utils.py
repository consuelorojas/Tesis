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

def min_in_interval(series, array, interval=15):
    '''
    Find the minimum value in the given series.

    Parameters:
    -----------
    series (pandas.Series):
      The input series.
    array (array-like):
      The array of indices to search.

    Returns:
    --------
    
    '''
    min1 = []
    min2 = []
    for i in array:
        min1.append(series[i-interval:i].min())
        min2.append(series[i:i+interval].min())

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

    num1 =  np.abs(df_general['Gradient Phase'].iloc[x])/2
    subset = np.abs(df_general['Gradient Phase'].iloc[x:y])
    num2 = find_nearest(subset, num1)

    indices2 = subset[np.abs(subset == num2)].index[0]
    indices_tau.append(indices2)
  return indices_tau

