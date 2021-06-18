import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import random
import pickle
import itertools

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
        activation = tf.nn.sigmoid(activation)
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
    
    def __init__(self, n, latentdim, e_activ='sigmoid', d_activ='sigmoid', k_reg=None, act_reg=None, adjacency=None, alpha=0.2, beta=10, deep_dims=None, learning_rate=0.025):
        super(Autoencoder, self).__init__()
        self.n = n
        self.latentdim = latentdim
        self.k_reg = k_reg
        self.history = {}
        self.alpha = alpha
        self.beta = beta
        self.adjacency = adjacency
        self.learning_rate=learning_rate
        self.laplacian = None
        if adjacency is not None:
            self.B = tf.cast(tf.convert_to_tensor(adjacency_to_loss(adjacency, self.beta)), dtype=tf.float32)
            self.laplacian = tf.cast(tf.convert_to_tensor(create_laplacian(adjacency)), dtype=tf.float32)
        
        initializer = tf.keras.initializers.RandomUniform(0, 0.01)
        
        self.encoder = tf.keras.Sequential()
        self.decoder = tf.keras.Sequential()
        
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(n,), dtype=tf.float32))
        
        if deep_dims is not None:
            for dim in deep_dims:
                self.encoder.add(layers.Dense(dim, activation=e_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, activity_regularizer=act_reg, dtype=tf.float32))
            for dim in deep_dims[::-1]:
                self.decoder.add(layers.Dense(dim, activation=d_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, dtype=tf.float32))
            
        self.encoder.add(layers.Dense(latentdim, activation=e_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, activity_regularizer=act_reg, dtype=tf.float32))
        
        self.decoder.add(layers.Dense(n, activation=d_activ, kernel_initializer=initializer, bias_initializer=initializer, kernel_regularizer=k_reg, dtype=tf.float32))
        
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

def loss(model, X):
    mse = tf.keras.losses.MeanSquaredError()
    
    H = model.encoder(X)
    X_=model.decoder(H)
    diff = X-X_
    if model.laplacian is not None:
        loss = tf.math.square(tf.norm(tf.math.multiply(model.B, X-X_)))
        reg_term = tf.linalg.trace(tf.linalg.matmul(tf.transpose(H), tf.linalg.matmul(model.laplacian, H)))
        return loss + 2 * model.alpha * reg_term
    else:
        return mse(X,X_)
    
def train_step(loss, model, original, epoch):
    with tf.GradientTape() as tape:
        opt=tf.keras.optimizers.Adam(learning_rate=model.learning_rate)
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
    for epoch in range(epochs):
        for step, batch_features in enumerate(training_dataset):
            train_step(loss, model, batch_features, epoch)
    return model.history