import numpy as np
from datasets import load_dataset

ds = load_dataset("ylecun/mnist")

# Note: increasing layer number will cause accuracy to drop to 9.74% (Random)
input_size   = 784
hidden_size  = 10
n_hidden     = 2
output_size  = 10
epochs       = 2
batch_size = 32
learning_rate = 0.1


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

#For all epochs
for epoch in range(epochs):

    print(f"Epoch {epoch +1}")

    index = 0
    shuffled_train = ds["train"].shuffle(seed=epoch)

    #For each example in the training set
    for example in shuffled_train:
        
        index += 1
        if index % 1000 == 0:
            print(f"Index: {index}")
        
        #Create input array and first activation
        inputValue = (np.array(example["image"]).flatten()/255.0).reshape(1,784)

        #logits lists all pre-activation values 
        logits = []
        #activations stores inputs and outputs for each layer
        activations = [inputValue]

        #forward pass (make sure to store logits and activations)
        for layer_idx in range(0, len(weights)):
            logit = inputValue @ weights[layer_idx] + biases[layer_idx]
            logits.append(logit)
                
            #Only last layer will use softmax instead of sigmoid so all outputs sum to 1
            if layer_idx < len(weights) - 1:
                output = sigmoid(logit)
            else:
                output = softmax(logit)

            activations.append(output)
            inputValue = output

        #Create expected output (row vector)
        one_hot = np.zeros(10)
        one_hot[example["label"]] = 1.0
        one_hot = one_hot.reshape(1, -1)

        # delta for output layer: predicted - actual
        delta = activations[-1] - one_hot

        for layer_idx in reversed(range(len(weights))):
            # activation of previous layer (1, prev_size)
            a_prev = activations[layer_idx]
            # (prev_size, 1) @ (1, cur_size) -> (prev_size, cur_size)
            dW = a_prev.T @ delta
            # shape (cur_size,)
            db = delta.reshape(-1)

            # gradient descent update
            weights[layer_idx] -= learning_rate * dW
            biases[layer_idx]  -= learning_rate * db

            # compute delta for next (previous in network) layer, as long as not at input layer
            if layer_idx > 0:
                # propagate through weights and multiply by sigmoid'(activation_of_layer_{l-1})
                delta = (delta @ weights[layer_idx].T) * sigmoid_derivative(activations[layer_idx])

#Evaluation on test set
correct = 0
total = 0

for example in ds["test"]:
    total += 1

    # forward pass only (same logic as training forward pass), just logits and activations not needed
    inputValue = (np.array(example["image"]).flatten() / 255.0).reshape(1, 784)
    for layer_idx in range(len(weights)):
        logit = inputValue @ weights[layer_idx] + biases[layer_idx]
        if layer_idx < len(weights) - 1:
            inputValue = sigmoid(logit)
        else:
            inputValue = softmax(logit)

    # predicted label is the largest of the softmax output
    predicted_label = int(np.argmax(inputValue, axis=-1)[0])
    actual_label = int(example["label"])

    # update counters
    if predicted_label == actual_label:
        correct += 1

# final percentages
percent_correct = (correct / total) * 100.0 if total else 0.0
percent_incorrect = 100.0 - percent_correct

print(f"Test results: {correct}/{total} correct")
print(f"Percentage correct: {percent_correct:.2f}%")
print(f"Percentage incorrect: {percent_incorrect:.2f}%")