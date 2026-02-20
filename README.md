# AI-Learning-Work
My AI Projects I have worked on.

ycleun-mnist Classification: Neural Network to train on 28x28 grayscale images of digits from the mnist data set.
Contains early non batching neural network and current batching example. Also has a Sketch Box to send 28x28 images to 
run inference on. NOTE: Sending sketches via Sketch Box results in lower accuracy than running on mnist testing samples 
as sketches may not necessarily be drawn by the user to the same likeliness as the training data.

Arduino Digit Classification: Takes the ycleun-mnist Classification Neural Network and modifies it to run on an Arduino.
It only runs the inference, and contains an additional Sketch Box to draw and send Digits to the Arduino to run inference on.

text-8 Transformer: Transformer based on the text-8 dataset which contains only lowercase words and no punctuation.
NOTE: This is still a work in progress.
