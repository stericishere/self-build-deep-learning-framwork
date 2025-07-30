# My Deep Learning Framework

A minimalist deep learning framework built from scratch using only NumPy, demonstrating core machine learning algorithms and neural network concepts.

## 🚀 Overview

This framework is my personal implementation of fundamental deep learning concepts, built to deepen my understanding of how neural networks work under the hood. It provides a clean, modular architecture for building and training neural networks without relying on high-level ML libraries.

## 🧠 What I've Implemented

### Core Components

- **🔢 Tensor Operations** (`tensor.py`): Custom tensor abstraction built on NumPy
- **🧬 Neural Networks** (`nn.py`): Feedforward neural network implementation with automatic differentiation
- **🏗️ Layer Architecture** (`layers.py`): Modular layer system including:
  - Linear (Dense) layers with weight and bias parameters
  - Activation functions (Tanh, ReLU)
  - Extensible base layer class for custom implementations
- **📉 Loss Functions** (`loss_function.py`): Mean Squared Error with gradient computation
- **⚡ Optimizers** (`optim.py`): Stochastic Gradient Descent (SGD) implementation
- **🎯 Training Loop** (`train.py`): Complete training pipeline with epoch management
- **📊 Data Handling** (`data.py`): Batch iteration and data shuffling utilities

### Key Algorithms Mastered

1. **Forward Propagation**: Efficient computation through network layers
2. **Backpropagation**: Automatic gradient computation using chain rule
3. **Gradient Descent**: Parameter optimization using computed gradients
4. **Batch Processing**: Mini-batch training for improved convergence

## 🎮 Demo Applications

### FizzBuzz Neural Network (`fizzbuzz.py`)
- **Problem**: Classic FizzBuzz game solved using neural networks
- **Approach**: Binary encoding of numbers (1-1024) to predict FizzBuzz categories
- **Architecture**: 10 → 50 → 4 network with Tanh activations
- **Innovation**: Demonstrates how neural networks can learn arithmetic patterns

### XOR Problem (`xor.py`)
- **Problem**: Non-linearly separable XOR function
- **Architecture**: 2 → 2 → 2 network with Tanh activations
- **Significance**: Classic test of neural network's ability to learn non-linear relationships

## 🏗️ Framework Architecture

```
Framework Structure:
├── Core Engine
│   ├── tensor.py      # Tensor operations
│   ├── nn.py         # Neural network class
│   └── layers.py     # Layer implementations
├── Training System
│   ├── train.py      # Training loop
│   ├── loss_function.py # Loss computations
│   └── optim.py      # Optimization algorithms
├── Data Pipeline
│   └── data.py       # Batch processing
└── Applications
    ├── fizzbuzz.py   # FizzBuzz solver
    └── xor.py        # XOR problem
```

## 💻 Quick Start

### Basic Neural Network
```python
from nn import NeuralNet
from layers import Linear, Tanh
from train import train
import numpy as np

# Create a simple network
net = NeuralNet([
    Linear(input_size=2, output_size=4),
    Tanh(),
    Linear(input_size=4, output_size=1)
])

# Prepare data
inputs = np.random.randn(100, 2)
targets = np.random.randn(100, 1)

# Train the network
train(net, inputs, targets, num_epochs=1000)
```

### Custom Training Configuration
```python
from optim import SGD
from loss_function import MSE
from data import BatchIterator

train(net, 
      inputs, 
      targets,
      num_epochs=5000,
      iterator=BatchIterator(batch_size=16, shuffle=True),
      loss=MSE(),
      optimizer=SGD(lr=0.001))
```

## 🔬 Technical Highlights

### Mathematical Foundations
- **Chain Rule Implementation**: Precise gradient computation through network layers
- **Matrix Operations**: Efficient linear algebra using NumPy
- **Activation Functions**: Smooth differentiable functions (Tanh, ReLU)

### Software Engineering
- **Modular Design**: Clean separation of concerns
- **Type Annotations**: Full Python typing for better code clarity
- **Extensible Architecture**: Easy to add new layers, optimizers, and loss functions

### Performance Features
- **Batch Processing**: Vectorized operations for training efficiency
- **Memory Management**: Efficient gradient storage and computation
- **Flexible Training**: Customizable training parameters

## 🎯 Learning Outcomes

Through building this framework, I've gained deep understanding of:

1. **Neural Network Fundamentals**
   - How gradients flow through networks
   - Parameter initialization strategies
   - Activation function properties

2. **Optimization Theory**
   - Gradient descent mechanics
   - Learning rate sensitivity
   - Convergence behavior

3. **Software Architecture**
   - Designing extensible ML systems
   - Managing computational graphs
   - Clean code practices in ML

4. **Mathematical Implementation**
   - Translating theory to code
   - Numerical stability considerations
   - Efficient matrix operations

## 🚀 Future Enhancements

### Planned Algorithms
- [ ] Adam optimizer
- [ ] Convolutional layers
- [ ] LSTM/RNN implementations
- [ ] Cross-entropy loss
- [ ] Dropout regularization
- [ ] Batch normalization

### Advanced Features
- [ ] GPU acceleration
- [ ] Model serialization
- [ ] Automatic differentiation engine
- [ ] Visualization tools
- [ ] More complex architectures

## 🛠️ Requirements

- Python 3.7+
- NumPy

## 🧪 Running the Examples

```bash
# Train XOR neural network
python xor.py

# Train FizzBuzz classifier
python fizzbuzz.py
```

## 📚 Educational Value

This framework serves as an educational tool for understanding:
- How popular frameworks like PyTorch/TensorFlow work internally
- The mathematical foundations of deep learning
- Best practices in ML software engineering
- The relationship between theory and implementation

## 📄 License

This project is for educational purposes and personal learning.

---

*This framework represents my journey in understanding machine learning from first principles. Each component was implemented to deepen my knowledge of the underlying algorithms and mathematical concepts that power modern AI systems.* 