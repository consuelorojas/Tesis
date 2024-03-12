import numpy as np
from tqdm import tqdm

def create_data(data, seq_len):
    N = len(data)
    X, Y = [], []
    for i in range(N-seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

def split_data(x,y, ratio):
    N = len(x)
    Ntrain = int(N * ratio)

    # x_train, y_train, x_test, y_test
    return x[:Ntrain], y[:Ntrain], x[Ntrain:], y[Ntrain:]

def create_batches(x,y, batch_size):
    N = len(x)
    batches = []
    for i in range(0,N,batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        batches.append((x_batch,y_batch))
    return batches

def train_model(model,
               criterion,
               optimizer,
               x_train,
               x_test,
               y_train,
               y_test,
               epochs=500):
    
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    
    for epoch in tqdm(range(epochs)):
        
        # put default model grads to zero
        optimizer.zero_grad()
        
        # predict the output
        pred = model(x_train)
        
        # calculate the loss 
        error = criterion(pred,y_train)
        
        # backpropagate the error
        error.backward()
        
        # update the model parameters
        optimizer.step()
        
        # save the losses 
        train_loss[epoch] = error.item()
        
        # test loss 
        test_pred = model(x_test)
        test_error = criterion(y_test,test_pred)
        test_loss[epoch] = test_error.item()
        
        if (epoch+1) % 5 ==0:
            print('Epoch :{}    Train Loss :{}    Test Loss :{}'.format((epoch+1)/epochs, error.item(), test_error.item()))
            
    return train_loss, test_loss