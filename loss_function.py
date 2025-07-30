"""
loss function meassures how good or bad a model is at making predictions.
we can use loss function to guide the model to make better predictions.
"""
import numpy as np
from tensor import Tensor
    
class Loss():
    """
    Base class for all loss functions.
    All loss functions must implement the loss and gradient methods.
    The loss method should return a scalar loss value.
    The gradient method should return a tensor of the same shape as the predicted tensor.
    The gradient is the partial derivative of the loss with respect to the predicted tensor.
    """
    def loss(self, predicted:Tensor, actual:Tensor) -> float:
        raise NotImplementedError
    
    def gradient(self, predicted:Tensor, actual:Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    Mean Squared Error loss function.
    """
    def loss(self, predicted:Tensor, actual:Tensor) -> float:
        return np.mean((predicted - actual) ** 2)
    
    def gradient(self, predicted:Tensor, actual:Tensor) -> Tensor:
        return 2 * (predicted - actual)