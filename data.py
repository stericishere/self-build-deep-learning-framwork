"""
We'll feed inputs into the neural network in btaches
So here some tools for iterating over data in batches.
"""
from typing import List, Iterator, NamedTuple
from tensor import Tensor
import numpy as np
from tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs:Tensor, targets:Tensor) -> Iterator[Batch]:
        """
        Yield batches of data.
        """
        raise NotImplementedError


class BatchIterator():
    def __init__(self, batch_size: int = 32, shuffle:bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs:Tensor, targets:Tensor) -> Iterator[Batch]:
        """
        Yield batches of data.
        """
        assert len(inputs) == len(targets)
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
            
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
