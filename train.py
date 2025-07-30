from nn import NeuralNet
from optim import SGD, Optimizer
from data import DataIterator, BatchIterator
from tensor import Tensor
from typing import Tuple, Iterator
from loss_function import Loss, MSE

def train(net:NeuralNet, 
          inputs:Tensor, 
          targets:Tensor, 
          num_epochs: int = 50, 
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(lr=0.1)
          ) -> None:
    """
    Train the network on the inputs and targets.
    """
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.gradient(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(f"Epoch {epoch} loss: {epoch_loss}")