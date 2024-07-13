# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 23:41:36 2024

@author: Marcel
"""



import numpy as np
import os
import pickle
import time
import gc
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
import random
import matplotlib.pyplot as plt

# Função que mostra gráfico
def show_graph_metric_loss(history_train, history_valid, prefix='', metric_name='F1-Score', save_path=None):
    # Training epochs and steps in x ticks
    total_epochs_training = len(history_train['loss'])
    x_ticks_step = 5
    
    # Create Figure
    plt.figure(figsize=(15,6))
    
    # There are 2 subplots in a row
    
    # X and X ticks for all subplots
    x = list(range(1, total_epochs_training+1)) # Could be length of other parameter too
    x_ticks = list(range(0, total_epochs_training+x_ticks_step, x_ticks_step))
    x_ticks.insert(1, 1)
    
    # First subplot (Metric)
    plt.subplot(1, 2, 1)
    
    plt.title(f'{prefix}{metric_name} per Epoch', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 19}) # Title, with font name and size
    
    plt.xlabel('Epochs', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) # Label of X axis, with font
    plt.ylabel(f'{metric_name}', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) # Label of X axis, with font
    
    plt.plot(x, history_train['f1']) # Plot Train Metric
    plt.plot(x, history_valid['f1']) # Plot Valid Metric
    
    plt.ylim(bottom=0, top=1) # Set y=0 on horizontal axis, and for maximum y=1
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) # Set y ticks
    plt.xticks(x_ticks) # Set x ticks
    
    plt.legend(['Train', 'Valid'], loc='upper left', fontsize=12) # Legend, with position and fontsize
    plt.grid(True) # Create grid
    
    # Second subplot (Loss)
    plt.subplot(1, 2, 2)
    
    plt.title(f'{prefix}Loss per Epoch', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 19})
     
    plt.xlabel('Epochs', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) 
    plt.ylabel('Loss', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 17}) 
    
    plt.plot(x, history_train['loss'])
    plt.plot(x, history_valid['loss'])
    
    
    plt.ylim(bottom=0, top=1) 
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(x_ticks)
    
    plt.legend(['Train', 'Valid'], loc='upper right', fontsize=12)
    plt.grid(True) 
    
    # Adjust layout
    plt.tight_layout()
    
    # Show or save plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)
        

class PrecRecF1:
    def __init__(self):
        self.ACTUAL_POSITIVE = 0 
        self.PREDICTED_POSITIVE = 0
        self.TP = 0        
        
    def update_state(self, y_true, y_pred):
        # Update statistics for calculus
        self.ACTUAL_POSITIVE += y_true.sum().item()
        self.PREDICTED_POSITIVE += y_pred.sum().item()
        self.TP += (y_true*y_pred).sum().item()
        
    def result(self):
        precision = self.TP / self.PREDICTED_POSITIVE if self.PREDICTED_POSITIVE > 0 else 0
        recall = self.TP / self.ACTUAL_POSITIVE if self.ACTUAL_POSITIVE > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def reset_state(self):
        self.ACTUAL_POSITIVE = 0 
        self.PREDICTED_POSITIVE = 0  
        self.TP = 0
    

    

def transform_augment_2arg(x, y):
    # Sorteia opção
    lista_opcoes = [1, 2, 3, 4, 5, 6, 7]
    opcao = random.choice(lista_opcoes)
    
    # Decide opção
    # Espelhamento Vertical (Flip)
    if opcao == 1:
        x = torch.flip(x, dims=(1,))
        y = torch.flip(y, dims=(1,))
        return x, y
    # Espelhamento Horizontal (Mirror)
    elif opcao == 2:
        x = torch.flip(x, dims=(2,))
        y = torch.flip(y, dims=(2,))
        return x, y
    # Rotação 90 graus
    elif opcao == 3:
        x = torch.rot90(x, k=1, dims=(1, 2))
        y = torch.rot90(y, k=1, dims=(1, 2))
        return x, y
    # Rotação 180 graus
    elif opcao == 4:
        x = torch.rot90(x, k=2, dims=(1, 2))
        y = torch.rot90(y, k=2, dims=(1, 2))
        return x, y
    # Rotação 270 graus
    elif opcao == 5:
        x = torch.rot90(x, k=3, dims=(1, 2))
        y = torch.rot90(y, k=3, dims=(1, 2))
        return x, y
    # Espelhamento Vertical e Rotação 90 graus
    elif opcao == 6:
        x = torch.rot90(torch.flip(x, dims=(1,)), k=1, dims=(1, 2))
        y = torch.rot90(torch.flip(y, dims=(1,)), k=1, dims=(1, 2))
        return x, y
    # Espelhamento Vertical e Rotação 270 graus
    elif opcao == 7:
        x = torch.rot90(torch.flip(x, dims=(1,)), k=3, dims=(1, 2))
        y = torch.rot90(torch.flip(y, dims=(1,)), k=3, dims=(1, 2))
        return x, y  


class EarlyStopper:
    def __init__(self, model, model_path, early_stopping_epochs, early_stopping_delta, early_stopping_on_metric):
        self.model = model
        self.model_path = model_path
        
        self.early_stopping_epochs = (early_stopping_epochs if early_stopping_epochs is not None else float('inf'))
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_on_metric = early_stopping_on_metric
        
        self.no_improvement_count = 0
        self.valid_param_best_model = (1e-20 if self.early_stopping_on_metric else float('inf'))

    def early_stop(self, valid_param):
        # Early Stopping on Metric
        if self.early_stopping_on_metric:
            diff = valid_param - self.valid_param_best_model
            # Metric stable - Increase Early Stopping
            if abs(diff) < self.early_stopping_delta:
                self.no_improvement_count = self.no_improvement_count + 1
                print(f'Early Stopping Count Increased to: {self.no_improvement_count}')
                if self.no_improvement_count >= self.early_stopping_epochs:
                    print('Early Stopping reached')
                    return True
            # Metric decreasing - Increase Early Stopping
            elif diff < 0:
                self.no_improvement_count = self.no_improvement_count + 1
                print(f'Early Stopping Count Increased to: {self.no_improvement_count}')
                if self.no_improvement_count >= self.early_stopping_epochs:
                    print('Early Stopping reached')
                    return True
            # Metric increasing - Save Model and Reset Early Stopping Count
            else:
                torch.save(self.model.state_dict(), self.model_path)
                self.no_improvement_count = 0
                self.valid_param_best_model = valid_param
                print('Model performance increasing, saving model')
        # Early Stopping on Loss   
        else:    
            diff = valid_param - self.valid_param_best_model
            # Loss stable - Increase Early Stopping
            if abs(diff) < self.early_stopping_delta:
                self.no_improvement_count = self.no_improvement_count + 1
                print(f'Early Stopping Count Increased to: {self.no_improvement_count}')
                if self.no_improvement_count >= self.early_stopping_epochs:
                    print('Early Stopping reached')
                    return True
            # Loss decreasing - Save Model and Reset Early Stopping Count
            elif diff < 0:
                torch.save(self.model.state_dict(), self.model_path)
                self.no_improvement_count = 0
                self.valid_param_best_model = valid_param
                print('Model performance increasing, saving model')
            # Loss increasing - Increase Early Stopping
            else:
                self.no_improvement_count = self.no_improvement_count + 1
                print(f'Early Stopping Count Increased to: {self.no_improvement_count}')
                if self.no_improvement_count >= self.early_stopping_epochs:
                    print('Early Stopping reached')
                    return True




class ModelTrainer:
    best_model_filename = 'best_model.pth'
    early_stopping_delta = 0.01 # Delta in relation to best result for training to continue
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )     
    def __init__(self, x_dir, y_dir, output_dir, model):
        # Directories
        self.x_dir = x_dir # Dir with X data
        self.y_dir = y_dir # Dir with Y data
        self.output_dir = output_dir # Dir to save Output data
        
        self.model = model # Model 
        
        self.model_path = output_dir + self.best_model_filename # Path to save model
        
    def _set_dataloaders(self, batch_size=32, shuffle=True):
        # Load Numpy Arrays and set them in necessary shape
        x_train = np.moveaxis(np.load(self.x_dir + 'x_train.npy'), 3, 1).astype(np.float32)
        y_train = np.moveaxis(np.load(self.y_dir + 'y_train.npy'), 3, 1).squeeze()
        
        x_valid = np.moveaxis(np.load(self.x_dir + 'x_valid.npy'), 3, 1).astype(np.float32)
        y_valid = np.moveaxis(np.load(self.y_dir + 'y_valid.npy'), 3, 1).squeeze()
        
        # Build Datasets from Tensors constructed form Numpy Arrays
        train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
        
        # Build Dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
        
        # Clean Memory
        del x_train, y_train, x_valid, y_valid, train_dataset, valid_dataset
        gc.collect()
        
    def _train_epoch(self, model, loss_fn, optimizer, main_metric_name, data_augmentation,
                     divisor_batch):
        assert main_metric_name in ('precision', 'recall', 'f1'), "main_metric_name must be one of options: " \
                                                                  "'precision', 'recall', 'f1'"
        
        train_loss_epoch = 0 # Variable to store the average loss of the epoch
        
        metrics = PrecRecF1() # Initialize Object that computes Metrics
        
        num_batches = len(self.train_dataloader) # Number of batches in one epoch
        size = len(self.train_dataloader.dataset) # Number of elements of the dataset
        model.train() # Set the model in training mode
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device) # Send variables to device (cpu or gpu)
            
            if data_augmentation:
                # Compute new "batches" 
                x_batches_train_augmented = []
                y_batches_train_augmented = []
                y.unsqueeze_(1) 
                for _ in range(divisor_batch-1):
                    list_transform = list(map(transform_augment_2arg, X, y))
                    x_batch_train_augmented, y_batch_train_augmented = torch.stack([l0[0] for l0 in list_transform]), \
                                                                       torch.stack([l1[1] for l1 in list_transform])
                    x_batches_train_augmented.append(x_batch_train_augmented)
                    y_batches_train_augmented.append(y_batch_train_augmented)
            
                # Concatenate original batch with new "batches"
                X = torch.concat([X] + x_batches_train_augmented, axis=0) 
                y = torch.concat([y] + y_batches_train_augmented, axis=0)
                
                # Squeeze y dimension of channels (only one channel)
                y.squeeze_(dim=1)
                
                # Delete computed variables
                del x_batch_train_augmented, y_batch_train_augmented, x_batches_train_augmented[:], y_batches_train_augmented[:]
                gc.collect()
            
            
            '''
            # DEBUG Transforms
            y.unsqueeze_(1)
            index_show = 0
            X_flip = torch.flip(X, dims=(2,)) # Flip
            X_mirror = torch.flip(X, dims=(3,)) # Mirror
            X_rot90 = torch.rot90(X, k=1, dims=(2, 3)) # Rotation 90
            X_rot180 = torch.rot90(X, k=2, dims=(2, 3)) # Rotation 180
            X_rot270 = torch.rot90(X, k=3, dims=(2, 3)) # Rotation 270
            X_flip_rot90 = torch.rot90(torch.flip(X, dims=(2,)), k=1, dims=(2, 3)) # Flip and rotation 90
            X_flip_rot270 = torch.rot90(torch.flip(X, dims=(2,)), k=3, dims=(2, 3)) # Flip and rotation 270
            
            
            plt.imshow(torch.moveaxis(X[index_show], 0, 2)); plt.axis('off'); plt.title('Original Image')
            plt.imshow(torch.moveaxis(X_flip[index_show], 0, 2)); plt.axis('off'); plt.title('Flipped Image')
            plt.imshow(torch.moveaxis(X_mirror[index_show], 0, 2)); plt.axis('off'); plt.title('Mirrored Image')
            plt.imshow(torch.moveaxis(X_rot90[index_show], 0, 2)); plt.axis('off'); plt.title('Rotated 90 Image')
            plt.imshow(torch.moveaxis(X_rot180[index_show], 0, 2)); plt.axis('off'); plt.title('Rotated 180 Image')
            plt.imshow(torch.moveaxis(X_rot270[index_show], 0, 2)); plt.axis('off'); plt.title('Rotated 270 Image')
            plt.imshow(torch.moveaxis(X_flip_rot90[index_show], 0, 2)); plt.axis('off'); plt.title('Flipped and Rotated 90 Image')
            plt.imshow(torch.moveaxis(X_flip_rot270[index_show], 0, 2)); plt.axis('off'); plt.title('Flipped and Rotated 270 Image')
            '''
         
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y.long())
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update loss acumulator
            loss_value = loss.item()
            train_loss_epoch = train_loss_epoch + loss_value    
    
            # Log every 200 batches 
            if batch % 200 == 0:
                loss, current = loss_value, (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
            # Define Y True and Y Pred
            y_pred = pred.argmax(dim=1)
            y_true = y
            
            # Update metrics object
            metrics.update_state(y_true, y_pred)
        
        # Average loss for the epoch and metrics for the epoch
        train_loss_epoch = train_loss_epoch/num_batches
        precision, recall, f1 = metrics.result()
        metrics_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        
        print(f"Train average loss for epoch: {train_loss_epoch:0.3f}")
        print(f"Train precision over epoch: {precision:0.4f}")
        print(f"Train recall over epoch: {recall:0.4f}")
        print(f"Train f1 over epoch: {f1:0.4f}")
                
        return train_loss_epoch, metrics_dict
    
    def _validation_epoch(self, model, loss_fn, main_metric_name):
        assert main_metric_name in ('precision', 'recall', 'f1'), "main_metric_name must be one of options: " \
                                                                  "'precision', 'recall', 'f1'"
        
        valid_loss_epoch = 0 
        
        metrics = PrecRecF1()
        
        num_batches = len(self.valid_dataloader) 
        model.eval() # Set the model in evaluation mode
        with torch.no_grad():
            for X, y in self.valid_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                
                # Compute and add loss to summation
                loss_value = loss_fn(pred, y.long()).item()
                valid_loss_epoch = valid_loss_epoch + loss_value
                
                # Define Y True and Y Pred
                y_pred = pred.argmax(dim=1)
                y_true = y
                
                # Update metrics object
                metrics.update_state(y_true, y_pred)
                # precision_until_here, recall_until_here, f1_until_here =  metrics.result()
                
    
        valid_loss_epoch /= num_batches
        precision, recall, f1 = metrics.result()
        metrics_dict = {'precision': precision, 'recall': recall, 'f1': f1}
        
        print(f"Validation average loss for epoch: {valid_loss_epoch:0.3f}")
        print(f"Validation precision over epoch: {precision:0.4f}")
        print(f"Validation recall over epoch: {recall:0.4f}")
        print(f"Validation f1 over epoch: {f1:0.4f}")
        
        return valid_loss_epoch, metrics_dict
    
    def _train_model(self, model, epochs, early_stopping_epochs, 
                     early_stopping_on_metric, main_metric_name, 
                     learning_rate, loss_fn, optimizer,  
                     data_augmentation, divisor_batch):
        '''
        Train a model with specified parameters.

        Parameters
        ----------
        model : model instance
            Model instance to train.
        epochs : int
            Number of epochs to train.
        early_stopping_epochs : int
            Number of early stopping epochs used. Set it to None to not use of early stopping.
        early_stopping_on_metric : bool
            If True, use early stopping based on a metric; else it is based on loss.
        main_metric_name : str
            Name of the metric to use for early stopping, if early_stopping_on_metric=True, 
            and to be saved in the history along with the loss, wich is used to do the training graphic.
        learning_rate : float
            Learning rate to be used on training.
        loss_fn : loss object 
            Loss object to be used on training.
        optimizer : optimizer object
            Optimizer object to be used on training.
        data_augmentation : bool
            Do or not data augmentation by divisor_batch. If true, it does data augmentation, else it doesn't.
        divisor_batch : int
            The input batch size is divided by this number, and the result is the number of elements in the 
            batch that is maintained. The remaining elements are generated by a random transform of the 
            maintained. Only exact division is accepted. This transform has 7 options: flip, mirror, 
            rotate 90, 180 or 270 degrees, flip and rotate 90 degrees, flip and rotate 270 degrees.

        Returns
        -------
        None.

        '''
        # Loss and chosen metric will be stored in a list, and the other metrics are stored in a list also
        history_train = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
        history_valid = {'loss': [], 'precision': [], 'recall': [], 'f1': []}
        
        # Create Early Stopping Object
        early_stopper = EarlyStopper(model=model, model_path=self.model_path,
                                     early_stopping_epochs=early_stopping_epochs, 
                                     early_stopping_delta=self.early_stopping_delta,
                                     early_stopping_on_metric=early_stopping_on_metric)        
        
        # Train model
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch+1))
            start_time_epoch = time.time()
            train_loss_epoch, train_metrics_epoch = self._train_epoch(model=model, loss_fn=loss_fn,
                                                                      optimizer=optimizer, 
                                                                      main_metric_name=main_metric_name,
                                                                      data_augmentation=data_augmentation,
                                                                      divisor_batch=divisor_batch)
            valid_loss_epoch, valid_metrics_epoch = self._validation_epoch(model=model, loss_fn=loss_fn,
                                                                           main_metric_name=main_metric_name)
            
            # Add to history
            history_train['loss'].append(train_loss_epoch); history_train['precision'].append(train_metrics_epoch['precision'])
            history_train['recall'].append(train_metrics_epoch['recall']); history_train['f1'].append(train_metrics_epoch['f1'])
            history_valid['loss'].append(valid_loss_epoch); history_valid['precision'].append(valid_metrics_epoch['precision'])
            history_valid['recall'].append(valid_metrics_epoch['recall']); history_valid['f1'].append(valid_metrics_epoch['f1'])   

            # Show time spent for the epoch
            end_time_epoch = time.time()
            print(f"Time taken for epoch: {(end_time_epoch-start_time_epoch)/60:.0f} min")            
    
            # Apply Early Stopping if applicable 
            valid_param = (valid_metrics_epoch[main_metric_name] if early_stopping_on_metric else valid_loss_epoch)
            if early_stopper.early_stop(valid_param):             
                break
            
        return history_train, history_valid            
        
    def train_with_loop(self, epochs=2000, early_stopping_epochs=25,
                        early_stopping_on_metric=True, main_metric_name='f1',
                        learning_rate=0.001, loss_fn=CrossEntropyLoss(), 
                        optimizer_class: type = Adam, 
                        batch_size=32, shuffle=True, data_augmentation=False,
                        divisor_batch=2):
        # Dictionary of parameters used when method is invoked
        dict_parameters = locals().copy()
        del dict_parameters['self']
        
        # divisor_batch * maintained_batch_size = batch_size
        # Here we consider the batch size as the final batch size, after data augmentation on the batch
        if data_augmentation:
            assert batch_size % divisor_batch == 0, "Batch size must be divisible by the augment " \
                                                    "factor of the batch when doing data augmentation"
            batch_size = batch_size // divisor_batch
            
        # Set datasets
        self._set_dataloaders(batch_size=batch_size, shuffle=shuffle)
        
        # Get model
        model = self.model.to(self.device)
        
        # Optimizer and learning rate
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        
        # Compute total time to train. Begin to count
        start_time_train = time.time()
        
        # Train the model
        history_train, history_valid = self._train_model(model=model, epochs=epochs, early_stopping_epochs=early_stopping_epochs,
                                                         early_stopping_on_metric=early_stopping_on_metric, 
                                                         main_metric_name=main_metric_name, learning_rate=learning_rate,
                                                         loss_fn=loss_fn, optimizer=optimizer, data_augmentation=data_augmentation,
                                                         divisor_batch=divisor_batch)    
        
        # End the train time count
        end_time_train = time.time()
        
        # Save history info in text and pickle file
        with open(os.path.join(self.output_dir, 'history_info.txt'), 'w') as f:
            f.write('Resultado = \n')
            f.write(str(history_train)+'\n')
            f.write(str(history_valid)+'\n')
            f.write(f'\nTempo total gasto no treinamento foi de {end_time_train-start_time_train} segundos, '
                    f'{(end_time_train-start_time_train)/3600:.1f} horas.\n')
            f.write(str(dict_parameters) + '\n')            
            
        with open(os.path.join(self.output_dir, f'history_pickle_{self.best_model_filename}.pickle'), "wb") as fp: 
            pickle.dump((history_train, history_valid), fp)
            
        # Save output in plot
        show_graph_metric_loss(history_train, history_valid, 
                               metric_name=main_metric_name.capitalize(),
                               save_path=self.output_dir+'plotagem.png')





























   


        

