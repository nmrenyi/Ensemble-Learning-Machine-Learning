# 2018011423 Ren Yi

import torch
from torch import optim
from sklearn.utils import shuffle
import numpy as np
class MLP(torch.nn.Module):
    '''
    MLP model in torch.nn.Module
    '''
    def __init__(self, max_feature):
        super(MLP, self).__init__()
        self.fc = torch.nn.Linear(max_feature, 1)
    def forward(self, x):
        return self.fc(x)

class MyMLP(object):
    '''
    MyMLP model 
    with "fit" and "predict" API
    can handle sparse matrix input
    '''
    def __init__(self, epoch = 6, lr = 1e-4, bs = 64, max_feature = 10000):
        '''
        configure the parameters
        '''
        self.epoch = epoch
        self.lr = lr
        self.bs = bs
        self.max_feature = max_feature
        self.net = MLP(max_feature)
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.MSELoss(reduce=True, size_average=True)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

    def loader(self, length, bs = 64):
        '''
        mini-batch loader, like Dataloader in pytorch
        but it can handle sparse matrix
        '''
        arr = shuffle(np.arange(length))
        start = 0
        indexs = []
        while True:
            indexs.append(arr[start:min(start + bs, length)])
            if start + bs  >= length:
                break
            start += bs
        return indexs

    def fit(self, train_matrix, train_label):
        '''
        train process
        '''
        for epoch in range(self.epoch):  # loop over the dataset multiple times
            for i, index in enumerate(self.loader(train_matrix.shape[0])):
                inputs, labels = torch.tensor(train_matrix[index].toarray()).float(), torch.tensor(np.array(train_label)[index]).float()
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(dim=1))
                loss.backward()
                self.optimizer.step()
        return self
    def predict(self, test_matrix):
        '''
        give a prediction based on input
        '''
        return self.net(torch.tensor(test_matrix.toarray()).float()).detach().numpy().ravel()
