# -*- coding: utf-8 -*-
"""
Classes for training Bayesian neural networks with Monte-Carlo Dropout.

Mehdi Anhichem (based on Yarin Gal (2015)).
University of Liverpool
12/11/2023
"""

#==============================================================================
# IMPORT REQUIRED MODULES
#==============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import keras

from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense, LeakyReLU, ReLU, PReLU
from tensorflow.keras import Model
from tensorflow.keras import optimizers

import random 
random.seed(123)

#==============================================================================
# CLASS BNN WITH MC DROPOUT
#==============================================================================

class BNN_MCDropout:
    def __init__(self, X_train, y_train, normalisation, name,
                 X_valid, y_valid,
                 hidden_layers, 
                 dropout = 0.05, reg_length = 1.0, tau = 1.0,
                 lr = 1e-3, loss_fct = 'mean_squared_error'):
        ''' 
        Constructor for the class creating a Bayesian Droupout Neural Network model trained with
        the probabilistic back propagation method. The Functional API from Keras is used.
        '''
        ''' Inputs:
            X_train:        matrix array of the shape (N,D) made of
                            training points.
            y_train:        array of the shape (N,1) made of target points.
            normalisation:  boolean determining input features normalisation.
            name:           string of model's name.
            X_valid:        matrix array of the shape (N,D) made of
                            validation points.
            y_valid:        array of the shape (N,1) made of target validation points.
            n_hidden:       list of number of neurons for each hidden layer. Its length is therefore the
                            number of hidden layers.
            n_epochs:       number of epochs to train the model.
            learning_rate:  learning rate of the Adam optimiser.
            dropout:        dropout rate for all the dropout layers in the network.
            reg_length:     lengthscale used for regularisation.
            tau:            tau value used for regularisation.
        '''
        self.name = name
        self.normalisation = normalisation
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        # Normalisation
        if normalisation == True:
            self.X_train_std = np.std(X_train, axis = 0)
            self.X_train_std[self.X_train_std == 0] = 1
            self.X_train_mean = np.mean(X_train, axis = 0)
            self.X_train = (self.X_train - np.full(self.X_train.shape, self.X_train_mean)) /np.full(self.X_train.shape, self.X_train_std)
            
            self.y_train_mean = np.mean(self.y_train)
            self.y_train_std = np.std(self.y_train)
            self.y_train = (self.y_train - self.y_train_mean) / self.y_train_std
            
            self.X_valid = (X_valid - np.full(X_valid.shape, self.X_train_mean)) / np.full(X_valid.shape, self.X_train_std)
            self.y_valid = (self.y_valid - self.y_train_mean) / self.y_train_std
            
        self.X_train = np.array(self.X_train, ndmin = 2)
        self.y_train = np.array(self.y_train, ndmin = 2)
        self.X_valid = np.array(self.X_valid, ndmin = 2)
        self.y_valid = np.array(self.y_valid, ndmin = 2)
        
        # Define the regulariser
        N = self.X_train.shape[0]
        reg = (reg_length**2) * (1 - dropout) / (2.0 * N * tau)
        
        # Input layer
        inputs = Input(shape=(self.X_train.shape[1]))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(hidden_layers[0], kernel_regularizer=l2(reg))(inter)
        inter = ReLU()(inter)
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(hidden_layers[i+1], kernel_regularizer=l2(reg))(inter)
            inter = ReLU()(inter)
            
        # Output layer
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(self.y_train.shape[1], kernel_regularizer=l2(reg))(inter)
        
        # Model compilation
        self.model = Model(inputs, outputs)
        self.model.compile(loss = loss_fct,
                           optimizer = optimizers.Adam(learning_rate = lr))
        
    def fit(self, n_epochs = 100, batch_size = 128):
        ''' 
        Function for training a Bayesian Dropout Neural Network model.
        '''
        ''' Inputs:
            n_epochs:       number of epochs to train the model.
            batch_size:     number of samples per batch. 
        '''
        self.batch_size = batch_size
        return self.model.fit(self.X_train, self.y_train, epochs = n_epochs, batch_size = self.batch_size, validation_data = (self.X_valid, self.y_valid))
    
    def predict(self, X_test, n_mc_samples = 100):
        ''' 
        Function for making predictions with the Bayesian Dropout Neural Network model.
        '''
        ''' Inputs:
            X_test:         matrix array of the shape (N,D) made of
                            testing points.
            n_mc_samples:   number of Monte Carlo estimate.
        '''
        # Normalisation
        if self.normalisation == True:
            X_test = (X_test - np.full(X_test.shape, self.X_train_mean)) / np.full(X_test.shape, self.X_train_std)
        X_test = np.array(X_test, ndmin = 2)
        # MC Dropout
        model = self.model
        Yt_hat = np.array([model.predict(X_test, batch_size = self.batch_size, verbose=0) for _ in range(n_mc_samples)])
        if self.normalisation == True:
            Yt_hat = Yt_hat * self.y_train_std + self.y_train_mean
        MC_mean = np.mean(Yt_hat, axis = 0)
        MC_std = np.std(Yt_hat, axis = 0)
        
        return (MC_mean, MC_std)
    


    def predict_parallel(self, X_test, n_mc_samples=100):
        ''' 
        Function for making predictions with the Bayesian Dropout Neural Network model.
        Uses parallel implementation based on MPI.
        '''
        ''' Inputs:
            X_test:         matrix array of the shape (N,D) made of
                            testing points.
            n_mc_samples:   number of Monte Carlo estimate.
        '''
        from mpi4py import MPI
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Normalisation
        if self.normalisation:
            X_test = (X_test - np.full(X_test.shape, self.X_train_mean)) / np.full(X_test.shape, self.X_train_std)
        X_test = np.array(X_test, ndmin=2)
        # MC Dropout
        model = self.model
        local_samples = n_mc_samples // size
        # Scatter the number of samples to each process
        local_samples = comm.scatter([local_samples] * size, root=0)
        # Perform local predictions
        local_predictions = [model.predict(X_test, batch_size=self.batch_size, verbose=0) for _ in range(local_samples)]
        # Gather all predictions to rank 0
        all_predictions = comm.gather(local_predictions, root=0)
        if rank == 0:
            # Combine predictions from all processes
            Yt_hat = np.concatenate(all_predictions)
            if self.normalisation:
                Yt_hat = Yt_hat * self.y_train_std + self.y_train_mean
            MC_mean = np.mean(Yt_hat, axis=0)
            MC_std = np.std(Yt_hat, axis=0)
            return MC_mean, MC_std
        else:
            return None