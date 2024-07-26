# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:05:37 2023

@author: marcel.rotunno
"""

import torch
from pathlib import Path
import pdb

# %% Limit GPU Memory

fraction_limit = 0.9
torch.cuda.set_per_process_memory_fraction(fraction_limit, 0)
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory
print(f'Total GPU memory available is {total_memory}. Limit to a fraction of {fraction_limit}.') 


# %% Data for Script

batch_size = 16
model_type = 'unetr'
early_stopping_epochs = 25

# %% Diretórios de entrada e saída

x_dir = r'entrada/'
y_dir = r'y_directory/'
output_dir = fr'saida_{model_type}_loop_1x_{batch_size}b_torch/'

if not Path(output_dir).exists():
    Path(output_dir).mkdir(exist_ok=True)


# %%  Importa para treinamento

from package_doc_pytorch.treinamento.functions_train import ModelTrainer
from package_doc_pytorch.treinamento.arquiteturas.unetr_modified import UNETR


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

# %% Build model

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = UNETR(config_dict).to(device)

# %% Create object for Train and Train

model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=output_dir, model=model)
model_trainer.train_with_loop(batch_size=batch_size)


                                      
                                      
                                      
                                      


