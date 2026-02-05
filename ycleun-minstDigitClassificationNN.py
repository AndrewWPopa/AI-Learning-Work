import numpy as np
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")

input_size   = 784
hidden_size  = 10
n_hidden     = 25
output_size  = 10
epochs       = 20

#Declare weights and biases list
weights = []
biases = []

#Input first hidden layer
weights.append(np.random.randn(input_size, hidden_size) * 0.01)
biases.append(np.zeros(hidden_size))

#Hidden layer next hidden layer (24 more times)
for _ in range(n_hidden - 1):
    weights.append(np.random.randn(hidden_size, hidden_size) * 0.1)
    biases.append(np.zeros(hidden_size))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

#For amount of epochs
for epoch in range(epochs):

    #For each example in the training set
    for example in ds["train"]:
        
        #Create Input Array and First passValue
        inputValue = (np.array(example["image"]).flatten()/255.0).reshape(1,784)
        #Logits lists all outputs before activation functions
        logits = []

        #for each layer in the weights matrix
        for layer_idx in range(0, len(weights)):

            #Logit is the value after weights and bias applied but before activation function
            logit = inputValue @ weights[layer_idx] + biases[layer_idx]
            logits.append(logit)

            #Activate with sigmoid except for last layer which uses softmax
            if layer_idx < len(weights) - 1:
                output = sigmoid(logit)
            else:
                output = softmax(logit)

            inputValue = output

        #Create expected output
        one_hot = np.zeros(10)
        one_hot[example["label"]] = 1.0

        for layer_idx in reversed(range(len(weights))):
            W = weights[layer_idx]
            passValuePrev = logits[layer_idx]
