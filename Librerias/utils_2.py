import numpy as np
import torch
from tqdm import tqdm

def create_data(data, seq_len, output_dim=1):
    N = len(data)
    X, Y = [], []
    for i in range(N-seq_len):
        X.append(data[i:i+seq_len])
        if output_dim > 1:
            Y.append(data[i+seq_len:i+seq_len+output_dim])
        else:
            Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

def split_data(x,y,ratio):
    assert len(x) == len(y)
    N = len(x)
    x_train, x_test = x[:int(N*ratio)], x[int(N*ratio):]
    y_train, y_test = y[:int(N*ratio)], y[int(N*ratio):]
    return x_train, y_train, x_test, y_test

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


def rollingWindowPrediction(model, x_test, steps = 50):
    output = []
    N = x_test.shape()[1]

    with torch.no_grad():
        model.eval()
        for elem in tqdm(x_test):
            elem = elem.view(N)
            test_aux = []
            count = 0
            while count < steps:
                    pred = model(elem.view(1,N,1))
                    test_aux.append(pred[0,0].item())
                    elem = torch.cat((elem[1:], pred[0]))
                    count += 1

            output.append(test_aux)
    return output

all_test_pred = []