This project runs a Neural Network to classify 28x28 (grayscale 0-255 value) pixel drawings of numbers, based on the ylecun-minst data set. 
It requires pretrained weights (included), which have been trained by the ylecun-minstClassification.py project.

Steps to run this project:

1. Set up Arduino UNO R3 (2KB RAM, 32KB FLASH) to write to an LED screen as seen in https://www.instructables.com/LCD-1602-With-Arduino-Uno-R3/

2. Plug USB to computer and Arduino

3. Start Arduino IDE and load mnist_inference folder

4. Select what USB to connect to (ie: COM6)

5. This step may be skipped since weights are already included, but if you would like to include your own weights (for two layers), then 
with pretrained weights (model.json) from ylecun-minstClassification.py output, run the JsonWeightParser.py, and copy output,
pasting it into the mnist_inference.ino file in the Arduino IDE. Ensure that #define headers are not deleted.

6. Compile the program and send it to the Arduino

7. Pip install required dependencies for project, such as tkinter, Pillow, serial.

8. Run DrawSendArea.py, and draw the digit and press send to send the digit (as an array) to the Arduino to run inference with it. 
Results will also be sent back from the arduino to the draw box