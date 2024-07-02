# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:06:15 2024

@author: Marcel
"""

# %% Import Libraries

import torch
import torch.nn as nn

# %% Hyperparameters

IMG_SIZE = 256 # No caso isso seria equivalente ao tamanho do PATCH em uma CNN
NUM_CHANNELS = 3
PATCH_SIZE = 16 # Isso seria equivalente a um subpatch (talvez o tamanho de um filtro em uma CNN?)
NUM_PATCHES = (IMG_SIZE//PATCH_SIZE) ** 2

HIDDEN_DIM = 128 # Dimensão do Embedding  
NUM_LAYERS = 12 # Número de Layers (Encoders) no Transformer
NUM_HEADS = 8 # Número de Cabeças para Atenção MultiCabeça
MLP_DIM = 256 # Dimensão da rede MLP no final da arquitetura
DROPOUT = 0.1 # Taxa de Dropout no MLP do Transformer
# MAX_FILTERS = 512 # Maximum number of filter in the Decoder

# Como são 4 deconvoluções, e partimos do valor do patch size, então
# vejo que em cada deconvolução, o patch duplica de tamanho.
# Com isso, ao final o tamanho da deconvolução precisa bater com
# o da imagem. O tamanho da deconvolução é patch_size*2*2*2*2,
# ou patch_size*2^4 ou patch_size*16

# %% Hyperparameters Dictionary

config_dict = {'IMG_SIZE': IMG_SIZE,
               'NUM_CHANNELS': NUM_CHANNELS,
               'PATCH_SIZE': PATCH_SIZE,
               'NUM_PATCHES': NUM_PATCHES,
               'HIDDEN_DIM': HIDDEN_DIM,
               'NUM_LAYERS': NUM_LAYERS,
               'NUM_HEADS': NUM_HEADS, 
               'MLP_DIM': MLP_DIM,
               'DROPOUT': DROPOUT} # ,MAX_FILTERS': MAX_FILTERS

# %% Patcher Class

class Patcher(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input Shape (B, C, H, W)
        bs, c, h, w = x.shape
        
        # Unfold. Transform to (Batch, Flat Patch Size, Number of Patches)
        x = self.unfold(x)
        
        # Chage axis order (Batch, Number of Patches, Flat Patch Size)
        x = x.moveaxis(2, 1)
        
        return x

# %% Conv and Deconv Classes

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding='same'):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        return self.deconv(x)

# %% UNETR Class

class UNETR_2D(nn.Module):
    def __init__(self, cf):
        super().__init__()
        self.cf = cf
        
        """ Extract patches """
        self.patcher = Patcher(cf["PATCH_SIZE"])
        
        """ Patch + Position Embeddings """
        self.patch_embed = nn.Linear(
            cf["PATCH_SIZE"]*cf["PATCH_SIZE"]*cf["NUM_CHANNELS"],
            cf["HIDDEN_DIM"]
        )
        
        self.positions = torch.arange(start=0, end=cf["NUM_PATCHES"], step=1, 
                                      dtype=torch.int32)
        self.pos_embed = nn.Embedding(cf["NUM_PATCHES"], cf["HIDDEN_DIM"])
        
        """ Transformer Encoder """
        self.trans_encoder_layers = []
        
        for i in range(cf["NUM_LAYERS"]):
            layer = nn.TransformerEncoderLayer(
                d_model=cf["HIDDEN_DIM"],
                nhead=cf["NUM_HEADS"],
                dim_feedforward=cf["MLP_DIM"],
                dropout=cf["DROPOUT"],
                activation=nn.GELU(),
                batch_first=True                
            )
            self.trans_encoder_layers.append(layer)
            
        """ CNN Decoder """
        ## Decoder 1
        self.d1 = DeconvBlock(cf["HIDDEN_DIM"], 512)
        self.s1 = nn.Sequential(
            DeconvBlock(cf["HIDDEN_DIM"], 512),
            ConvBlock(512, 512)            
        )
        self.c1 = nn.Sequential(
            ConvBlock(512+512, 512),
            ConvBlock(512, 512)
        )
        
        ## Decoder 2
        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(cf["HIDDEN_DIM"], 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256),
        )
        self.c2 = nn.Sequential(
            ConvBlock(256+256, 256),
            ConvBlock(256, 256)
        )
        
        ## Decoder 3
        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(cf["HIDDEN_DIM"], 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128)
        )
        self.c3 = nn.Sequential(
            ConvBlock(128+128, 128),
            ConvBlock(128, 128)
        )
        
        ## Decoder 4
        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.c4 = nn.Sequential(
            ConvBlock(64+64, 64),
            ConvBlock(64, 64)
        )
        
        """ Output """
        self.output = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        
    def forward(self, inputs):
        """ Extract patches """
        patches = self.patcher(inputs)
        
        """ Patch + Position Embeddings """
        patch_embed = self.patch_embed(patches) # [8, 256, 768]
        
        positions = self.positions
        pos_embed = self.pos_embed(positions)
        
        x = patch_embed + pos_embed
        
        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []
        
        for i in range(self.cf["NUM_LAYERS"]):
            layer = self.trans_encoder_layers[i]
            x = layer(x)
            
            if (i+1) in skip_connection_index:
                skip_connections.append(x)
                
        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections
        
        ## Reshaping
        batch = inputs.shape[0]
        z0 = inputs.view((batch, self.cf["NUM_CHANNELS"], self.cf["IMG_SIZE"], self.cf["IMG_SIZE"]))
        
        shape = (batch, self.cf["HIDDEN_DIM"], self.cf["PATCH_SIZE"], self.cf["PATCH_SIZE"])
        z3 = z3.view(shape)
        z6 = z6.view(shape)
        z9 = z9.view(shape)
        z12 = z12.view(shape)
        
        ## Decoder 1
        x = self.d1(z12)        
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        ## Decoder 2
        x = self.d2(x)        
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)
        
        ## Decoder 3
        x = self.d3(x)        
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)
        
        ## Decoder 4
        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)
        
        """ Output """
        output = self.output(x)
        
        return output

# %%

x = torch.rand(10, 3, 256, 256)
y = (torch.rand(10, 1, 256, 256) > 0.95).type(torch.uint8) # 5% of 1's

teste_unetr = UNETR_2D(config_dict)
saida_teste_unetr = teste_unetr(x)




