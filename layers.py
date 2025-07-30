"""
Our neural nets will be built using layers.
Each layer will pass an input tensor forward and 
produce an output tensor.

inputs -> Linear -> Tanh -> Linear -> output
Layers can have parameters which will be learned during training.
"""
# Standard library imports
import numpy as np
from tensor import Tensor
from typing import Callable, Dict

class Layer:
    """
    Base class for all layers.
    """
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.gradients: Dict[str, Tensor] = {}

    def forward(self, input:Tensor) -> Tensor:
        """
        Forward pass through the layer.
        """
        raise NotImplementedError
    
    def backward(self, output_grad:Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer.
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Computes output = input @ weights + bias.
    """
    def __init__(self, input_size:int, output_size:int) -> None:
        """
        inputs will be (batch_size, input_size)
        outputs will be (batch_size, output_size)
        """
        super().__init__()
        self.params["weights"] = np.random.randn(input_size, output_size)
        self.params["bias"] = np.random.randn(output_size)
        
    def forward(self, inputs:Tensor) -> Tensor:
        """
        output = input @ weights + bias
        """
        self.inputs = inputs
        return inputs @ self.params["weights"] + self.params["bias"]
    
    def backward(self, gradient:Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)
        
        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.gradients["weights"] = self.inputs.T @ gradient
        self.gradients["bias"] = np.sum(gradient, axis=0)
        return gradient @ self.params["weights"].T
 
F = Callable[[Tensor], Tensor]
 
class Activation(Layer):
    """
    An activation layer just applies an element-wise function 
    to its input.
    """
    def __init__(self, f: F, f_prime: F) -> None:
        """
        f is the activation function, f_prime is its derivative.
        """
        super().__init__()
        self.f = f
        self.f_prime = f_prime
        
    def forward(self, inputs:Tensor) -> Tensor:
        """
        Forward pass through the layer.
        """
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, gradient:Tensor) -> Tensor:
        """
        Backward pass through the layer.
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        and dy/dx = f'(x)
        and dy/dg = f'(x) * g'(z)
        
        g'(x) = f_prime(self.inputs)
        f'(x) = gradient
        """
        return self.f_prime(self.inputs) * gradient

# Activation functions
def tanh(x:Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x:Tensor) -> Tensor:
    return 1 - np.tanh(x) ** 2

def relu(x:Tensor) -> Tensor:
    return np.maximum(0, x)

def relu_prime(x:Tensor) -> Tensor:
    return (x > 0).astype(float)

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)