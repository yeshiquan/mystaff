# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
from torch.autograd import Variable

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../datasets/MNIST_data'

input_size = 28 * 28 # image size of MNIST data
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 1e-3

train_dataset = dsets.MNIST(root = DATA_PATH, 
                           train = True, 
                           transform = transforms.ToTensor(), 
                           download = False) 

test_dataset = dsets.MNIST(root = DATA_PATH, 
                           train = False, 
                           transform = transforms.ToTensor(), 
                           download = False) 

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
                                                  batch_size = batch_size, 
                                                  shuffle = True)  
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                  batch_size = batch_size,
                                                  shuffle = True)

class LRNet(nn.Module):
    def __init__(self,input_size, num_classes):
        super(LRNet, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

net = LRNet(input_size, num_classes)
print(net)


# 交叉熵损失
criterion = nn.CrossEntropyLoss()
# 随机梯度下降
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)

#print(train_loader)

for epoch in range(num_epochs):
#for epoch in range(2):
    print(' epoch = %d' % epoch)
    for i, (images, labels) in enumerate(train_loader): 
        '''
        用的是mini_batch的方式来训练梯度下降
        一次迭代加载batch_size个样本,然后进行矩阵运算
        '''
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        # 反向传播，求得每个参数的梯度
        loss.backward()
        # 更新一次参数
        optimizer.step()

        if  i % 100 == 0:
            print('current loss = %.5f' % loss.data[0])

    #print(np.shape(images))


# test the model
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    outputs = net(images)
    #torch.max(x, n) 沿着n维进行某种操作。得到的是某一维度的最大值之类的，如果不加维度n，则返回所有元素的最大值之类的
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('accuracy of the model %.2f' % (100 * correct / total))
