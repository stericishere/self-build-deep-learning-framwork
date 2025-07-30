import numpy as np
from nn import NeuralNet
from layers import Linear, Tanh
from train import train

inputs = np.array([
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]])

targets = np.array([
    [1, 0], 
    [0, 1], 
    [0, 1], 
    [1, 0]])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2),
    Tanh()
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(f"input: {x}, target: {y}, output: {predicted}")
