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

    #array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def  min_in_subset(df_general, index, interval):
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
    min1 = np.argmin(np.abs(df_general[index-interval:index]))
    min2 = np.argmin(np.abs(df_general[index:index+interval]))

    return min1+index-interval, min2+index
   

def min_in_arrays(df_general, serie, array, interval=15):
  """
  Find the minimum values within subsets of a DataFrame column for each element in an array.

  Parameters:
  -----------
  df_general (DataFrame):
    The DataFrame containing the data.
  serie (str):
    The name of the column in the DataFrame to analyze.
  array (list):
    The array of elements to create subsets and find minimum values.
  interval (int, optional):
    The size of the subsets. Defaults to 15.

  Returns:
  --------
  DataFrame:
    A DataFrame with the start and end values of the minimum subsets.
  """
  min1 = []
  min2 = []
  for elem in array:
    aux1, aux2 = min_in_subset(df_general[serie], elem, interval)
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
  serie = df_general['Gradient Phase']

  for i in range(len(array)):
    x = array[i]
    y = df_min['end'][i]

    num1 =  np.abs(serie[x])/2
    subset2 = np.abs(serie[x:y])
    print(x,y)
    num2 = find_nearest(subset2, num1)

    indices2 = subset2[np.abs(subset2 == num2)].index[0]
    indices_tau.append(indices2)
  return indices_tau

