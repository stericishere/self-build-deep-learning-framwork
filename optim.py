"""
we use a optimizer to adjust the parameters 
of the neural network base on the gradients 
computer during the backpropagation.
"""
from nn import NeuralNet

class Optimizer:
    """
    Base class for all optimizers.
    """
        
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError
    
class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    SGD use a single data point to update the parameters
    of the neural network by subtracting the gradient of the loss 
    with respect to the parameters from the parameters.
    w = w - learning_rate * gradient
    b = b - learning_rate * gradient
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        
    def step(self, net: NeuralNet) -> None:
        for param, gradient in net.params_n_gradients():
            param -= self.lr * gradient
            