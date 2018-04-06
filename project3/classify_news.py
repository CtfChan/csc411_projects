#!/usr/bin/python 

import random
import numpy as np
import math
import pickle
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


# Define Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, 30)
        self.linear2 = nn.Linear(30, 100)
        self.linear3 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        y_pred = self.linear3(h_relu2)
        return y_pred


#Reload model
input_size = 5832
num_classes = 2
model_L2 = LogisticRegression(input_size, num_classes)
model_L2.load_state_dict(torch.load('weights'))

#Reload data (deserialize)
with open('filename.pickle', 'rb') as handle:
    word_dict = pickle.load(handle)

#Grab data from file
some_data = sys.argv[1]
data = []
for line in open(some_data):
    l = line.rstrip('\n').split()
    data.append(l)


#generate matrix
num_unique_words = len(word_dict)
mat_set = np.zeros((len(data), num_unique_words))
for i in range(len(data)):
	for word in data[i]:
	 	#must check if word is in word_dict
	 	if word in word_dict: mat_set[i][word_dict[word]] = 1

#Feed to model
test_input = Variable(torch.from_numpy(mat_set), requires_grad=False).type(torch.FloatTensor)
prediction = model_L2(test_input).data.numpy()
results = np.argmax(prediction, 1)
for res in results:
    if res == 0:
        print("1")
    else:
        print("0")






