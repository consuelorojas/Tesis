#funciones auxiliares para el programa principal
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from earlystop import EarlyStopper
import torch.utils.data as data_utils
from scipy.signal import butter, filtfilt

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
  data2 =  np.delete(data, [drop_index], axis=1)
  xs, ys = [], []
  for i in range(len(data) - window - horizon + 1):
    x = data[i:i+window]
    y = data2[i+window: i+window+horizon]
    xs.append(x)
    ys.append(y)

  return np.array(xs), np.array(ys)

def subsample(data, n):
  """
  Subsamples the given data by a factor of n.

  Parameters:
  -----------
  data (numpy.ndarray):
    The input data.
  n (int):
    The subsampling factor.

  Returns:
  -----------
  numpy.ndarray:
    The subsampled data.

  Examples:
  -----------
  >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  >>> subsample(data, 2)
  array([1, 3, 5, 7, 9])
  >>> subsample(data, 3)
  array([1, 4, 7, 10])
  """
  
  subset = data[::n]

  #filtro para eliminar frecuencias parasitarias
  order = 4
  cutoff = 100

  b,a = butter(order, cutoff, fs = 1000, btype = 'low', analog = False, output='ba')
  df = filtfilt(b,a, subset) 
  return df

# nn functions for dataset creation

def standarize_data(data):
  """
  Standarizes the given data.

  Parameters:
  -----------
  data (numpy.ndarray):
    The input data.

  Returns:
  -----------
  numpy.ndarray:
    The standarized data.
  """
  mean = np.mean(data)
  std = np.std(data)
  return (data - mean) / std

def split_data(dataset, split=0.8):
  """
  Splits the given dataset into training and testing sets based on the specified split ratio.

  Parameters:
  -----------
  dataset (list):
    The dataset to be split.
  split (float):
    The ratio of the dataset to be used for training. Default is 0.8.

  Returns:
  -----------
  tuple: A tuple containing the training set and testing set.
  """

  train_size = int(len(dataset) * split)
  train, test = dataset[:train_size], dataset[train_size:]
  return train, test



def create_dataset(data, lookback):
  """
  Create a dataset for time series forecasting.

  Parameters:
  -----------
    dataset (numpy.ndarray):
      The input dataset.
    lookback (int):
      The number of previous time steps to use as input for each sample.

  Returns:
  -----------
    torch.Tensor:
      The input data tensor.
    torch.Tensor:
      The target data tensor.
  """
  X, y = [], []
  dataset = standarize_data(data)
  for i in range(len(dataset) - lookback):
    X.append(dataset[i:(i + lookback)])
    y.append(dataset[i + 1:i + lookback + 1])
  X = np.array(X)
  y = np.array(y)
  if dataset.ndim == 1:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
  else:
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def create_loader(X, y, batch_size=32, shuffle=True):
  loader = data_utils.DataLoader(data_utils.TensorDataset(X, y), shuffle = shuffle, batch_size = batch_size)
  return loader
# creck points

def checkpoint(model, optimizer, filename):
    torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
    }, filename)
    
def resume(model, optimizer, filename):
  check_point  = torch.load(filename)
  model.load_state_dict(check_point['model'])
  optimizer.load_state_dict(check_point['optimizer'])


def checkpoint_plot(epoch, avg_train_losses, avg_val_losses, train_loss, y_pred, y_val):
  print(f'Epoch: {epoch}, Train Loss: {avg_train_losses[-1]}, Test Loss: {avg_val_losses[-1]}')
  print(f'Epoch: {epoch}, Loss: {train_loss[-1]}')
  plt.plot(y_pred[-1].numpy(), label = 'Prediction')
  plt.plot(y_val[-1].numpy(), label = 'Real')
  plt.title('Prediction vs Real epoch: ' + str(epoch))
  plt.legend()
  plt.show()
  plt.close()

  plt.plot(avg_train_losses, label='Train Loss')
  plt.plot(avg_val_losses, label='Validation Loss')
  plt.title('Losses train vs validation epoch: ' + str(epoch))
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()
  plt.close()


# training and testing functions

def train_model(model, optimizer, criterion, train_loader, val_loader, n_epochs, ncheckpoint =10):
  avg_train_losses, avg_val_losses = [], []
  early_stopper = EarlyStopper(patience=ncheckpoint, min_delta=1**-10)

  for epoch in tqdm(range(n_epochs)):
    train_loss, val_loss = [], []
    
    try: 
      model.train()
      for x_batch, y_batch in train_loader:
        y_pred = model(x_batch.float())
        loss = criterion(y_pred, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

      #validation
      model.eval()
      with torch.no_grad():
        for x_val, y_val in val_loader:
          y_pred = model(x_val.float())
          loss = criterion(y_pred, y_val.float())
          val_loss.append(loss.item())
      
      avg_train_losses.append(np.average(train_loss))
      avg_val_losses.append(np.average(val_loss))

      

      if epoch % ncheckpoint == 0:
        checkpoint(model, optimizer, f'checkpoint_{epoch}.pth')
        checkpoint_plot(epoch, avg_train_losses, avg_val_losses, train_loss, y_pred, y_val)

      if early_stopper.early_stop(avg_val_losses[-1]):
        checkpoint(model, optimizer, f'earlystop_{epoch}.pth')
        print(f'Early stopping, saving checkpoint of {epoch}')
  
    except KeyboardInterrupt:
      print('\nTraining interrupted by user')
      break  

  return model, avg_train_losses, avg_val_losses

      

def predictions(model, loader):
  model.eval()
  with torch.no_grad():
    for x_val, y_val in loader:
      y_pred = model(x_val.float())

  return y_pred


