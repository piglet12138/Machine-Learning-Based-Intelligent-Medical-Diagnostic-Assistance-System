import torch
import random
import pandas as pd
import numpy as np
from torch.utils import data
from torch import nn
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class Data(data.Dataset):

    def __init__(self, X , y ):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def mlp_train(mlp , batch_size = 10 , num_epochs = 50 , lr = 0.001 , X = None, y =None, wd = 0.3 , X_test = None, y_test= None):

    for param in mlp.parameters():
        param.data.normal_()

    X_mean = X.mean(axis = 0)
    X_std = X.std(axis = 0)
    X = (X - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std
    train_data = Data(X,y)

    dataloader = data.DataLoader(train_data, batch_size = batch_size , shuffle=True)
    loss = nn.CrossEntropyLoss(reduction='none')#weight=weights
    trainer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=0.01)
    loss_values = []
    accuracy = []
    train_acc =[]
    for epoch in range(num_epochs):
        running_loss = 0.0
        '''
        if epoch > 500 and abs(train_acc[-1]-train_acc[-2]) < 1e-10:
            print("stop in advance")
            break
        '''
        for XX, yy in dataloader:
            XX = XX.type(torch.float32)
            yy = yy.type(torch.long)
            trainer.zero_grad()
            l = loss(mlp(XX) ,yy)
            l.sum().backward()
            trainer.step()
            running_loss += l.sum().item()

        mlp.eval()

        prediction_test = mlp(torch.tensor(X_test).type(torch.float32)).detach().numpy().argmax(axis=1)
        cm = confusion_matrix(prediction_test, y_test)
        print(cm)
        acc = np.sum(prediction_test==y_test)/len(y_test)
        accuracy.append(acc)

        prediction = mlp(torch.tensor(X).type(torch.float32)).detach().numpy().argmax(axis=1)
        cm = confusion_matrix(prediction, y)
        t_acc = np.sum(prediction==y)/len(y)
        train_acc.append(t_acc)

        epoch_loss = running_loss / len(X)
        loss_values.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, test_acc: {acc:.4f}, train_acc: {t_acc:.4f}')

        mlp.train()
    best = max(accuracy)
    print(f"best accuracy is {best} in epoch {accuracy.index(best)}")
    '''
    
    plt.plot(train_acc, label='Train acc')
    plt.plot(accuracy, label='test acc')
    plt.title('accuracy')
    plt.ylim((0,1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    '''
    return prediction_test

