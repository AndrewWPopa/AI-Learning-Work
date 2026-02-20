import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# Load trained model
data = np.load("ylecun-minst Classification/mnist_model.npz", allow_pickle=True)
weights = [np.array(w, dtype=np.float64) for w in data["weights"]]
biases = [np.array(b, dtype=np.float64) for b in data["biases"]]

canvas_size = 280
grid_size = 28

root = tk.Tk()
root.title("Draw a Digit")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

image = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    r = 8  # brush radius

    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

canvas.bind("<B1-Motion>", paint)

def predict():
    # Resize 280x280 to 28x28
    small = image.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(small).astype(np.float32)
    
    np.savetxt("drawn_digit.txt", img_array, fmt="%03d")
    
    # Normalize
    img_array /= 255.0

    # Flatten
    img_array = img_array.reshape(1, 784)

    # Forward pass
    current = img_array

    for i in range(len(weights)):
        logit = current @ weights[i] + biases[i]
        if i < len(weights) - 1:
            current = 1 / (1 + np.exp(-logit))  # sigmoid
        else:
            exp = np.exp(logit - np.max(logit))
            current = exp / np.sum(exp)

    prediction = np.argmax(current)
    confidence = np.max(current)

    result_label.config(text=f"Prediction: {prediction} ({confidence:.2f})")

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
    result_label.config(text="")

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear)
clear_button.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack()

root.mainloop()
