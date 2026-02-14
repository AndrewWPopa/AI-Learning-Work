import random

def load_text8(path="text8"):
    with open(path, 'r') as f:
        text = f.read()
    return text

def encode_whole100M_dataset(text, encode):
    return encode(text)

def get_batch(encoded, block_size, batch_size):
    inputs = []
    targets = []

    for _ in range(batch_size):
        i = random.randint(0, len(encoded) - block_size - 1)

        x = encoded[i:i+block_size]
        y = encoded[i+1:i+block_size+1]

        inputs.append(x)
        targets.append(y)

    return inputs, targets
