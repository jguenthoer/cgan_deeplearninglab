import numpy as np
import torch
from torch import nn



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Blowup(nn.Module):
    """opposite of flatten
    go from linear to convolution
    batchx36864 -> batchx1024x6x6
    
    """
    def forward(self, input):
        return input.view(input.size(0), 256, 5, 4)

class Generator(nn.Module):
    def __init__(self, droprate = 0.5):


        super(Generator, self).__init__()


        self.linear1 = nn.Linear(40, 40)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(50, 256*4*5)
        self.batchn1 = nn.BatchNorm2d(256)
        self.transpose1 = nn.ConvTranspose2d(256, 128, 5, 2)
        self.batchn2 = nn.BatchNorm2d(128)
        self.transpose2 = nn.ConvTranspose2d(128, 128, 5, 2)
        self.batchn3 = nn.BatchNorm2d(128)
        self.transpose3 = nn.ConvTranspose2d(128, 64, 5, 2)
        self.batchn4 = nn.BatchNorm2d(64)
        self.transpose4 = nn.ConvTranspose2d(64, 32, 5, 2)
        self.batchn5 = nn.BatchNorm2d(32)
        self.transpose5 = nn.ConvTranspose2d(32, 16, 5, 2)
        self.batchn6 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 8, 4, 2)
        self.batchn7 = nn.BatchNorm2d(8)
        self.transpose6 = nn.ConvTranspose2d(8, 3, 4, 2, padding= (17,21))
        
        self.blowup = Blowup()
        
        self.LRelu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p = droprate)
        

    def forward(self, att, landmark):
        
        att = self.LRelu(self.linear1(att))
        landmark = self.LRelu(self.linear2(landmark))
        x = torch.cat((att,landmark), -1)
        x = self.LRelu(self.linear3(x))
        x = self.batchn1(self.blowup(x))
        x = self.LRelu(self.batchn2(self.transpose1(x)))
        x = self.dropout(x)
        x = self.LRelu(self.batchn3(self.transpose2(x)))
        x = self.dropout(x)
        x = self.LRelu(self.batchn4(self.transpose3(x)))
        x = self.dropout(x)
        x = self.LRelu(self.batchn5(self.transpose4(x)))
        x = self.dropout(x)
        x = self.LRelu(self.batchn6(self.transpose5(x)))
        x = self.LRelu(self.batchn7(self.conv1(x)))
        
        
        x = torch.tanh(self.transpose6(x))
        return x
        
 
class Discriminator(nn.Module):
    def __init__(self):


        super(Discriminator, self).__init__()


        self.conv1 = nn.Conv2d(3, 32, 5,2)
        self.batchn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.batchn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 2)
        self.batchn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 5, 2)
        self.batchn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 1, 5, 2)
        #self.lin1 = nn.Linear(128,128*10*8)

        
        self.LRelu = nn.LeakyReLU()
        self.flat = Flatten()

    def forward(self, x):
        
        #cond = self.LRelu(self.lin1(cond))
        #cond = cond.view(cond.size(0), 128, 10,8)
        x = self.LRelu(self.batchn1(self.conv1(x)))
        x = self.LRelu(self.batchn2(self.conv2(x)))
        x = self.LRelu(self.batchn3(self.conv3(x)))
        x = self.LRelu(self.batchn4(self.conv4(x)))
        #x = torch.cat((x,cond), -3)
        x = self.conv5(x)



        x = x.mean((-1,-2))
        
        return x    
        


                     
