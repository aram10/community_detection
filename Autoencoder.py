import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import random
import pickle
import itertools
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, losses
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.linalg import diag
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.generators.community import LFR_benchmark_graph
from itertools import count

from helpers import *

class SparseRegularizer(keras.regularizers.Regularizer):
    
    def __init__(self, rho = 0.01,beta = 1):
        """
        rho  : Desired average activation of the hidden units
        beta : Weight of sparsity penalty term
        """
        self.rho = rho
        self.beta = beta
        
    def __call__(self, activation):
        rho = self.rho
        beta = self.beta
        rho_bar = K.mean(activation, axis=0)
        rho_bar = K.maximum(rho_bar,1e-10) 
        KLs = rho*K.log(rho/rho_bar) + (1-rho)*K.log((1-rho)/(1-rho_bar))
        return beta * K.sum(KLs)
    
    def get_config(self):
        return {
            'rho': self.rho,
            'beta': self.beta
        }

class Autoencoder(Model):
    
    """
    Attributes:
        n: number of nodes in network
        latentdim: dimension of embedding
        e_activ: activation function for encoder
        d_activ: activation function for decoder
        k_reg: hidden layer weights regularizer
        act_reg: hidden layer activity regularizer
        adjacency: graph adjacency matrix
        beta: scalar multiplier for sparsity constraint, if any
        deep_dims: if not a single-layer autoencoder, a list of dimensions not including n and latentdim
        learning_rate: learning rate
        lam: scalar multiplier for temporal smoothness term, if doing dynamic CD
        loss: loss function
        subspace_distance: ordinal value for metric to compare latent space distances, if doing dynamic CD
            Let A (nxp), B (nxr) be subspaces in question.
            0: sqrt(max(p,r) - Tr(A*A^{T}*B*B^{T}))
            1: (1/sqrt(2)) * ||A*A^{T} - B*B^{T}||_{F}
        c: number of communities for this time step
        c_old: number of communities for previous time step
        full_rank: whether or not to consider entire latent space for temporal smoothness loss or only take largest c/c_old eigenvectors
        Z_old: community indicator matrix for previous time step
        fine_tuning: If training a deep model, whether or not to fine-tune all layers together with a low learning rate after pretraining
        fine_tuning_learning_rate: Learning rate for fine tuning
        fine_tuning_training_epochs: Number of epochs to fine tune for
    """
    
    def __init__(self, n, latentdim, e_activ='sigmoid', d_activ='sigmoid', k_reg=None, act_reg=None, adjacency=None, beta=10, deep_dims=None, learning_rate=0.025, lam=1, loss=tf.keras.losses.MeanSquaredError(), subspace_distance=0, c=-1, c_old=-1, full_rank=True, Z_old=None, fine_tuning=True, fine_tuning_learning_rate=0.0001, fine_tuning_training_epochs=10):
        super(Autoencoder, self).__init__()
        self.n = n
        self.latentdim = latentdim
        self.k_reg = k_reg
        self.history = {}
        self.beta = beta
        self.adjacency = adjacency
        self.learning_rate=learning_rate
        self.lam = lam
        self.H = None
        self.c = c
        self.c_old = c_old
        self.loss = loss
        self.full_rank = full_rank
        self.subspace_distance = subspace_distance
        self.Z_old = Z_old
        self.fine_tuning = fine_tuning
        self.fine_tuning_learning_rate = fine_tuning_learning_rate
        self.fine_tuning_training_epochs = fine_tuning_training_epochs
        
        initializer = tf.keras.initializers.RandomUniform(0, 0.01)
        
        self.encoder_layers = []
        self.decoder_layers = []
        
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        
        #add extra layers if deep network
        if deep_dims is not None:
            for dim in deep_dims:
                self.encoder_layers.append(layers.Dense(dim, activation=e_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, activity_regularizer=act_reg, dtype=tf.float32))
            for dim in deep_dims[::-1]:
                self.decoder_layers.append(layers.Dense(dim, activation=d_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, dtype=tf.float32))
            
        self.encoder_layers.append(layers.Dense(latentdim, activation=e_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, activity_regularizer=act_reg, dtype=tf.float32))
        
        self.decoder_layers.append(layers.Dense(n, activation=d_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, dtype=tf.float32))
        
        self.encoder = tf.keras.Sequential(self.encoder_layers)
        self.decoder = tf.keras.Sequential(self.decoder_layers)
               
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
    #embedding of last time step, if temporal
    def set_past_embedding(self, H):
        self.H = H;
        
    def weights_length(self):
        return len(self.encoder_layers)
    
    def freeze_all_layers(self):
        for layer in self.encoder_layers:
            layer.trainable = False
        for layer in self.decoder_layers:
            layer.trainable = False
                    
    def unfreeze_all_layers(self):
        for layer in self.encoder_layers:
            layer.trainable = True
        for layer in self.decoder_layers:
            layer.trainable = True
            
    def freeze_layer_pair(self, index):
        self.encoder_layers[index].trainable = False
        self.decoder_layers[(self.weights_length() - 1) - index].trainable = False     
        
    def unfreeze_layer_pair(self, index):
        self.encoder_layers[index].trainable = True
        self.decoder_layers[(self.weights_length() - 1) - index].trainable = True
        
        
def loss(model, X):    
    H = model.encoder(X)
    X_=model.decoder(H)
    #if we have an embedding for last timestep, use dynamic CD
    if model.H is not None and model.c != -1 and model.c_old != -1:
        K1 = tf.linalg.matmul(H, tf.transpose(H))
        w1, U1 = tf.linalg.eigh(K1)
        K2 = tf.linalg.matmul(model.H, tf.transpose(model.H))
        w2, U2 = tf.linalg.eigh(K2)
        #whether or not to consider only largest k eigenvectors, k=#communities
        if model.full_rank == False:
            w1 = w1[model.n - model.c:]
            U1 = U1[:,model.n - model.c:]
            w2 = w2[model.n - model.c_old:]
            U2 = U2[:,model.n - model.c_old:]
        P1 = tf.linalg.matmul(U1, tf.transpose(U1))
        P2 = tf.linalg.matmul(U2, tf.transpose(U2))
        #chordal distance
        if model.subspace_distance == 1:
            norm = tf.norm(tf.math.subtract(P1, P2))
            s_dist = 1/np.sqrt(2) * norm
            return model.loss(X,X_) + model.lam * s_dist
        #metric proposed in Zuccon et al., 2014
        else:
            s_dist = np.sqrt(max(model.c, model.c_old) - tf.linalg.trace(tf.matmul(P1, P2)))
            return model.loss(X,X_) + model.lam * s_dist
    elif model.Z_old is not None:
        K = tf.matmul(model.Z_old, tf.transpose(model.Z_old))
        s  = tf.matmul(tf.transpose(H), tf.matmul(K, H))
        return model.loss(X,X_) + model.lam * s
    else:
        return model.loss(X,X_)
    
def train_step(loss, model, original, epoch, fine_tuning=False):
    with tf.GradientTape() as tape:
        opt=tf.keras.optimizers.Adam(learning_rate=model.learning_rate)
        if(fine_tuning):
            opt=tf.keras.optimizers.Adam(learning_rate=model.fine_tuning_learning_rate)
        l = loss(model, original)
        model.history[epoch] = l
        gradients = tape.gradient(l, model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)
        
def train(model, epochs, batch_size, data):
    training_dataset = tf.data.Dataset.from_tensor_slices(data)
    training_dataset = training_dataset.cache()
    training_dataset = training_dataset.shuffle(data.shape[0])
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    if(model.weights_length() == 1):
        for epoch in range(epochs):
            for step, batch_features in enumerate(training_dataset):
                train_step(loss, model, batch_features, epoch)
    else:
        #layer-by-layer training of deep model
        model.freeze_all_layers()
        for x in range(model.weights_length()):
            model.unfreeze_layer_pair(x)
            for epoch in range(epochs):
                for step, batch_features in enumerate(training_dataset):
                    train_step(loss, model, batch_features, epoch)
            model.freeze_layer_pair(x)
        model.unfreeze_all_layers()
        if(model.fine_tuning):
            for epoch in range(model.fine_tuning_training_epochs):
                for step, batch_features in enumerate(training_dataset):
                    train_step(loss, model, batch_features, epoch, fine_tuning=True)
    return model.history