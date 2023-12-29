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

def get_indices_min(serie, indice, intervalo):
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

