import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, n_classes,layers,dropout=False,dropout_rate=0.6):

        super(Net, self).__init__()
        self.indim = input_size
        self.hidden = hidden_dim
        self.hdims = layers
        self.n_classes=n_classes
        self.dropout = dropout
        self.dropout_rate=dropout_rate
        current_dim = input_size
        self.layers = nn.ModuleList()
        for dimension in range(self.hdims):
            self.layers.append(nn.Linear(current_dim,self.hidden))
            current_dim = self.hidden
            if dropout:
                self.layers.append(nn.Dropout(p=self.dropout_rate))
        self.output=nn.Linear(self.hidden,n_classes)

    def forward(self, X):

        scores = Variable(torch.zeros((X.shape[0],self.n_classes)), requires_grad=False)
        for i in range(X.shape[0]):
            x = X[i]
            x = x.view(-1,self.indim)
            for j,l in enumerate(self.layers):
                x=F.elu(self.layers[j](x)+l(x))
            x = F.softmax(self.output(x), dim=1)
            #x = (self.output(x))
            scores[i]=x
        return scores

