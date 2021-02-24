#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:06:17 2021

@author: alejomonbar
"""
import numpy as np
import pennylane as qml
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# import pandas

# Load data and split into testing and training data

def normalize_data(data):
    """
    Data (n, 5)
    n = cases
    5 = input variables
    """
    max_data = np.max(data, axis = 0)
    min_data = np.min(data, axis = 0)
    normalized_data = (data - min_data)/ (max_data - min_data)
    return normalized_data

trainData = np.load("./Data/trainData.npy")
testData = np.load("./Data/testData.npy")


max_data = np.max(trainData, axis = 0)
min_data = np.min(trainData, axis = 0)
normalized_data = (trainData - min_data)/ (max_data - min_data)

np.random.shuffle(normalized_data)

normalized_input = normalized_data[:,:4]
normalized_output = normalized_data[:,4]

WIRES = 4
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3

def variational_circuit(params, x):
    for n, i in enumerate(x):            
        qml.RX(np.pi*i, wires=n)
    qml.templates.StronglyEntanglingLayers(params, wires=range(WIRES))
    return qml.expval(qml.PauliZ(0))

# def variational_circuit(params, x):
#     qml.templates.IQPEmbedding(x, wires=range(WIRES),n_repeats=3)
#     qml.templates.StronglyEntanglingLayers(params, wires=range(WIRES))
#     return qml.expval(qml.PauliZ(0))




def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data

    Args:
        inputs (array[float]): input data
        targets (array[float]): targets

    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]

def correlation_training(X_train, Y_train, Params=None):
    """Develop and train a Quantum neural network to create a correlation for 
    ANN-based correlation for frictional pressure drop of non-azeotropic 
    mixtures during cryogenic forced boiling.


    Args:
        X_train (np.ndarray): An array of floats of size (3929, 4) to be used as training data.
        Y_train (np.ndarray): An array of size (3929,) which is the frictional pressure
        drop normalized.
        Params (list): list of the parameters from a old optimization (to continue a training)
    Returns:
        params: (list len=epochs)Params of the circuit which produces the desired output
        Cost:(list len=epochs) Cost function for the different epochs
    """

    # Use this array to make a prediction for the labels of the data in X_test
    dev = qml.device("default.qubit", wires=WIRES)

    # Instantiate the QNode
    circuit = qml.QNode(variational_circuit, dev)

    def cost_f(params, x, y):
        suma = 0
        n = len(x)
        for i in range(n):
            y_c = (circuit(params, x[i]) + 1) / 2
            suma += (y_c - y[i]) ** 2
        # print(suma)
        return suma
    # Train using Adam optimizer and evaluate the classifier
    learning_rate = 0.1
    epochs = 30
    batch_size = 300
    
    opt = qml.optimize.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

    # Minimize the circuit
    # opt = qml.GradientDescentOptimizer(stepsize=0.1)
    if not isinstance(Params,list):
        params = np.random.random((LAYERS, WIRES, 3))
        Params = []
    else:
        params = Params[-1]
    Cost = [0]
    
    stop = 0
    for it in range(epochs):
        cost_sum = 0
        for Xbatch, ybatch in tqdm(iterate_minibatches(X_train, Y_train, batch_size=batch_size)):
            params, cost = opt.step_and_cost(lambda v: cost_f(v, Xbatch, ybatch), params)
            cost_sum += cost
        Params.append(params)
        Cost.append(cost_sum)
        print("Cost: ",cost_sum)
        if Cost[-2] - Cost[-1] < 0.1:
            stop += 1
        if stop > 2:
            break
        
    return Params, Cost

params, cost = correlation_training(normalized_input, normalized_output)


dev = qml.device("default.qubit", wires=WIRES)
circuit = qml.QNode(variational_circuit, dev)
Y_pred = []
for i in normalized_input:
    Y_pred.append((circuit(params[-1],i) + 1)/2)
Y_pred = np.array(Y_pred)

error = 100*np.sum(np.abs(Y_pred - normalized_output))/len(Y_pred)

normalized_test = normalize_data(testData)
normalized_test_i = normalized_test[:,:4]
normalized_test_o = normalized_test[:,4]

Y_test = []
for i in normalized_test_i:
    Y_test.append((circuit(params[-1],i) + 1)/2)
Y_test = np.array(Y_test)

error_test = 100*np.sum(np.abs(Y_test - normalized_test_o))/len(Y_test)
