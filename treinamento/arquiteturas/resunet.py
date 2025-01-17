# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:00:12 2024

@author: Marcel
"""

import torch
import torch.nn as nn




class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        
        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x
    
class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        """ Convolutional Layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride) # For stride=2, padding=1, side of the patch is even,      
                                                                                  # kernel_size=3, output has exactly the half of the original side
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1) # For stride=2, padding=1, side of the patch is even, 
                                                                              # kernel_size=3, output has exactly the half of the original side                                                                             
        """ Shotcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)
        
    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)        
        s = self.s(inputs)
        
        skip = x + s        
        return skip    

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.r = residual_block(in_c+out_c, out_c)       
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)        
        return x        
        
class ResUnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        """ Encoder 1 """
        self.c11 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.c13 = nn.Conv2d(3, 64, kernel_size=1, padding='same')
        
        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2)
        self.r3 = residual_block(128, 256, stride=2)
        
        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2)
        
        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        
        """ Output """
        self.output = nn.Conv2d(64, 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s 
        
        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)
        
        """ Bridge """
        b = self.r4(skip3)
        
        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)
        
        """ Output """
        output = self.output(d3)
        # output = self.softmax(output)
                
        return output
    
# %% Teste

class Teste(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.c = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        
    def forward(self, inputs):
        x = self.c(inputs)
        return x

x = torch.rand(10, 3, 256, 256)
y = (torch.rand(10, 3, 256, 256) > 0.95).type(torch.uint8) # 5% of 1's


x = torch.rand(10, 3, 8, 8)
y = (torch.rand(10, 3, 8, 8) > 0.95).type(torch.uint8) # 5% of 1's







teste_resunet = ResUnet()
saida_teste = teste_resunet(x)       