"""
A neural network is just a collection of layers.
"""
from tensor import Tensor
from typing import List, Iterator, Tuple
from layers import Layer

class NeuralNet:
    """
    A feedforward neural network.
    """
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers
    
    def forward(self, inputs:Tensor) -> Tensor:
        """
        Forward pass through the network.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, output_grad:Tensor) -> None:
        """
        Backward pass through the network.
        """
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
        return output_grad

    def params_n_gradients(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Return a list of all parameters and their gradients.
        """
        for layer in self.layers:
            for name, param in layer.params.items():
                gradient = layer.gradients[name]
                yield param, gradient
    