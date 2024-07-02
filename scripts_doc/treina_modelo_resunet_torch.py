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
model_type = 'resunet'
early_stopping_epochs = 25

# %% Diretórios de entrada e saída

x_dir = r'entrada/'
y_dir = r'y_directory/'
output_dir = fr'saida_{model_type}_loop_1x_{batch_size}b_torch/'

if not Path(output_dir).exists():
    Path(output_dir).mkdir(exist_ok=True)


# %%  Importa para treinamento

from package_doc_pytorch.treinamento.functions_train import ModelTrainer
from package_doc_pytorch.treinamento.arquiteturas.resunet import ResUnet

# %% Create object for Train and Train


model_trainer = ModelTrainer(x_dir=x_dir, y_dir=y_dir, output_dir=output_dir, model_class=ResUnet)
model_trainer.train_with_loop()


                                      
                                      
                                      
                                      


