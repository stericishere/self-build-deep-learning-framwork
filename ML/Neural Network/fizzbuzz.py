"""
FizzBuzz is the following problem:
For each of the numbers 1 to 100:
    if the number is divisible by 3, print "Fizz"
    if the number is divisible by 5, print "Buzz"
    if the number is divisible by both 3 and 5, print "FizzBuzz"
    otherwise, print the number
"""
import numpy as np
from nn import NeuralNet
from layers import Linear, Tanh
from train import train
from typing import List
from optim import SGD

def fizzbuzz_encode(x:int) -> List[int]:
    """
    Convert a number into a list of its binary digits.
    """
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]
    
def binary_encode(x:int) -> List[int]:
    """
    Convert a number into a list of its binary digits.
    """
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x:int) -> List[int]:
    """
    Convert a number into a list of its binary digits.
    """
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101,1024)
])

targets = np.array([
    fizzbuzz_encode(i)
    for i in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net, 
      inputs, 
      targets,
      num_epochs=5000,
      optimizer=SGD(lr=0.001))

for x in range(101, 1024):
    predicted = net.forward(binary_encode(x))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizzbuzz_encode(x)) #true label
    labels = [str(x), "Fizz", "Buzz", "FizzBuzz"]
    print(f"input: {x}, predicted: {labels[predicted_idx]}, actual: {labels[actual_idx]}")