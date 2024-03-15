import numpy as np
import torch
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import copy

def create_sequences(data, seq_len, output_dim=1):
    N = len(data)
    X, Y = [], []
    if output_dim == 1:
        for i in range(N-seq_len):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len])
    else:
        for i in range(N-seq_len-output_dim+1):
            X.append(data[i:i+seq_len])
            aux = data[i+seq_len:i+seq_len+output_dim]
            Y.append(aux)
    return np.array(X), np.array(Y)


def create_batches(x,y, batch_size):
    N = len(x)
    batches = []
    for i in range(0, N, batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        batches.append((x_batch,y_batch))
    return batches

def plot_loss(train_loss, test_loss, title):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def simpleTrain_model(model,
               criterion,
               optimizer,
               x_train,
               x_test,
               y_train,
               y_test,
               epochs=500):
    
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    best_model = copy.deepcopy(model)
    best_epoch = 0
    for epoch in tqdm(range(epochs)):
        try:
            model.train()
            # put default model grads to zero
            optimizer.zero_grad()
            
            # predict the output
            pred = model(x_train)
            #print(pred.size())
            
            # calculate the loss 
            error = criterion(pred.squeeze() ,y_train)
            
            # backpropagate the error
            error.backward()
            
            # update the model parameters
            optimizer.step()
            
            # save the losses 
            train_loss[epoch] = error.item()
            
            model.eval()
            # test loss

            test_pred = model(x_test)
            test_error = criterion(test_pred.squeeze(), y_test)
            test_loss[epoch] = test_error.item()

            if test_loss[epoch] < test_loss[epoch-1]:
                best_epoch = epoch
                best_model = copy.deepcopy(model)


            if (epoch+1) % 5 ==0:
                utils.checkpoint(model, optimizer, f'checkpoint_{epoch}.pth')
                plot_loss(train_loss[:epoch], test_loss[:epoch], 'Train and Test Loss')
                print('Epoch :{}    Train Loss :{}    Test Loss :{}'.format((epoch+1)/epochs, error.item(), test_error.item()))
                
            if utils.earlystop(test_loss[:epoch], 10, epoch):
                utils.checkpoint(best_model, optimizer, f'earlystop_{best_epoch}.pth')
                print('Early stopping at epoch: ', epoch)

        except KeyboardInterrupt:
            print('\nTraining Interrupted by user')
            break
        
    return train_loss, test_loss


#predicción tipo rolling window
def rollingWindowPrediction(model, x_test, steps = 50):
    output = []
    N = x_test.size()[-1]
    #print(N)

    with torch.no_grad():
        model.eval()
        for elem in tqdm(x_test):
            elem = elem.view(N).unsqueeze(0)
            #print(elem.shape)
            test_aux = []
            count = 0
            while count < steps:
                    pred = model(elem)
                    #print(pred.shape)
                    test_aux.append(pred[0,0].item())
                    elem = torch.cat((elem.squeeze(0)[1:], pred[0])).unsqueeze(0)
                    count += 1

            output.append(test_aux)
    return output
