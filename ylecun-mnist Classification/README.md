A Neural Network to classify 28x28 (grayscale 0-255 value) pixel drawings of numbers, based on the ylecun-minst data set.


Steps to run this project:

1. Download dependencies that are missing at the top of the library.

2. Run ylecn-mnistBatching.py


File Descriptions:

ylecun-mnistBatching.py: Current Neural Network which works using batching.
    layer_array: Each number in the array is a layer with the number of nodes denoted by the number. Change numbers and amount of numbers to adjust network width and length.

ylecun-mnistSingleExample.py: First Neural Network created which runs without batching. Outdated example, as the batching Neural Network is more efficient and accurate.

DrawBox.py: Starts a Sketch Box and then draw a digit to send to the Neural Network for inference. Uses weights generated from ylecun-mnistBatching.py

mnist_model.npz & model.json: Different formats to store the weights and biases.

drawn_digit.txt: 28x28 array of 1 byte integers (format 000) representing the drawn digit.

Forward Pass Explanation.jpg: An explanation of how a forward pass of the neural network works.