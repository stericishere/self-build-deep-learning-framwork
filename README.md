<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# SELF-BUILT ML LIBRARY

<em>A minimalist deep learning framework built from scratch using only NumPy</em>

<!-- BADGES -->
<img src="https://img.shields.io/badge/python-3.7+-blue.svg?style=flat&logo=python&logoColor=white" alt="python">
<img src="https://img.shields.io/badge/numpy-required-orange.svg?style=flat&logo=numpy&logoColor=white" alt="numpy">
<img src="https://img.shields.io/badge/machine%20learning-from%20scratch-green.svg?style=flat" alt="ml-from-scratch">
<img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat&logo=opensourceinitiative&logoColor=white" alt="license">

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

A comprehensive machine learning library built entirely from scratch using only NumPy, demonstrating fundamental ML algorithms and neural network concepts. This educational framework provides clean implementations of core deep learning components, making it perfect for understanding how popular frameworks like PyTorch and TensorFlow work internally.

**Key Philosophy**: Learn by building. Every component is implemented from first principles to provide deep understanding of the mathematical foundations that power modern AI systems.

---

## Features

🧠 **Neural Networks**
- Feedforward neural networks with automatic differentiation
- Modular layer architecture (Linear, Activation layers)
- Backpropagation implementation from scratch

⚡ **Training Infrastructure**
- SGD optimizer with customizable learning rates
- Mean Squared Error loss function
- Batch processing with data shuffling
- Complete training pipeline

🎯 **Machine Learning Algorithms**
- **Decision Trees**: Tree-based classification and regression
- **Random Forest**: Ensemble method with multiple decision trees
- **K-Nearest Neighbors**: Instance-based learning algorithm
- **Logistic Regression**: Linear classification with gradient descent
- **Naive Bayes**: Probabilistic classification
- **Linear Regression (Least Squares)**: Basic regression analysis

🔧 **Core Components**
- Custom tensor abstraction built on NumPy
- Gradient computation and automatic differentiation
- Modular and extensible architecture
- Type-annotated codebase for clarity

🎮 **Demo Applications**
- **XOR Problem**: Classic non-linear classification
- **FizzBuzz Neural Network**: Arithmetic pattern recognition
- **Gradient Checking**: Numerical verification of gradients

---

## Project Structure

```sh
└── Self-built ML library/
    ├── ML/
    │   ├── Decision Trees/
    │   │   ├── DecisionTree.py
    │   │   └── Random_forest.py
    │   ├── KNN/
    │   │   └── run_kn.pyn
    │   ├── logistic regression/
    │   │   ├── check_grad.py
    │   │   ├── logistic.py
    │   │   └── run_logistic_regression copy.py
    │   ├── Neural Network/
    │   │   ├── layers.py
    │   │   ├── nn.py
    │   │   ├── optim.py
    │   │   ├── train.py
    │   │   ├── fizzbuzz.py
    │   │   └── xor.py
    │   └── loss_function.py
    ├── Probability model/
    │   ├── LRLS.py
    │   └── naive_bayes.py
    ├── data.py
    └── tensor.py
```

### Project Index

<details open>
	<summary><b><code>SELF-BUILT ML LIBRARY/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b>tensor.py</b></td>
					<td style='padding: 8px;'>Defines <code>Tensor</code> as a type alias for NumPy's <code>ndarray</code>, providing a consistent interface for multi-dimensional array operations throughout the framework. This abstraction enables clear and maintainable tensor operations across all modules.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b>data.py</b></td>
					<td style='padding: 8px;'>Provides essential data iteration utilities for neural network training. Features a <code>BatchIterator</code> class that handles data shuffling and mini-batch generation, enabling efficient gradient descent training with configurable batch sizes.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- ML Submodule -->
	<details>
		<summary><b>ML</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ ML</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b>loss_function.py</b></td>
					<td style='padding: 8px;'>Implements the foundational loss function architecture with a base <code>Loss</code> class and Mean Squared Error (MSE) implementation. Provides both loss computation and gradient calculation methods essential for neural network optimization.</td>
				</tr>
			</table>
			<!-- Neural Network Submodule -->
			<details>
				<summary><b>Neural Network</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ ML.Neural Network</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>nn.py</b></td>
							<td style='padding: 8px;'>Core neural network implementation featuring the <code>NeuralNet</code> class that orchestrates forward and backward passes through layer sequences. Manages automatic gradient computation and parameter updates across the entire network architecture.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>layers.py</b></td>
							<td style='padding: 8px;'>Comprehensive layer architecture including base <code>Layer</code> class, <code>Linear</code> (dense) layers with weight/bias parameters, and activation functions (Tanh, ReLU). Implements forward propagation and backpropagation for each layer type with proper gradient computation.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>optim.py</b></td>
							<td style='padding: 8px;'>Optimization algorithms for training neural networks, featuring Stochastic Gradient Descent (SGD) implementation with configurable learning rates and parameter update mechanisms.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>train.py</b></td>
							<td style='padding: 8px;'>Complete training pipeline that coordinates the entire learning process, managing epochs, batch processing, forward/backward passes, and parameter updates. Provides a clean interface for training neural networks with customizable configurations.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>fizzbuzz.py</b></td>
							<td style='padding: 8px;'>Demonstrates neural network capabilities by solving the classic FizzBuzz problem using a 10→50→4 architecture. Shows how networks can learn arithmetic patterns through binary encoding and multi-class classification.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>xor.py</b></td>
							<td style='padding: 8px;'>Classic XOR problem implementation showcasing the network's ability to learn non-linearly separable functions. Uses a 2→2→2 architecture to demonstrate fundamental neural network capabilities.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- logistic regression Submodule -->
			<details>
				<summary><b>logistic regression</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ ML.logistic regression</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>logistic.py</b></td>
							<td style='padding: 8px;'>Logistic regression implementation with sigmoid activation, cross-entropy loss computation, and gradient-based optimization. Provides binary classification capabilities with performance evaluation metrics.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>run_logistic_regression copy.py</b></td>
							<td style='padding: 8px;'>Complete training script for logistic regression featuring gradient descent optimization, train/validation evaluation, gradient checking for correctness verification, and visualization of training progress through loss curves.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>check_grad.py</b></td>
							<td style='padding: 8px;'>Gradient verification utility that numerically validates analytical gradient computations, ensuring correctness of backpropagation implementations through finite difference approximation.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- Decision Trees Submodule -->
			<details>
				<summary><b>Decision Trees</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ ML.Decision Trees</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>DecisionTree.py</b></td>
							<td style='padding: 8px;'>Decision tree implementation featuring recursive tree building, information gain calculation, and prediction mechanisms for both classification and regression tasks. Handles feature selection and tree pruning strategies.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>Random_forest.py</b></td>
							<td style='padding: 8px;'>Random Forest ensemble method that combines multiple decision trees using bootstrap aggregating (bagging) and random feature selection. Improves prediction accuracy and reduces overfitting through ensemble voting.</td>
						</tr>
					</table>
				</blockquote>
			</details>
			<!-- KNN Submodule -->
			<details>
				<summary><b>KNN</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ ML.KNN</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b>run_kn.pyn</b></td>
							<td style='padding: 8px;'>K-Nearest Neighbors implementation featuring distance-based classification, k-value optimization, and various distance metrics. Provides instance-based learning for both classification and regression tasks.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- Probability model Submodule -->
	<details>
		<summary><b>Probability model</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ Probability model</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b>naive_bayes.py</b></td>
					<td style='padding: 8px;'>Naive Bayes classifier implementation using probabilistic approach with independence assumptions. Features Gaussian and multinomial variants for different data types, providing efficient classification with probability estimates.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b>LRLS.py</b></td>
					<td style='padding: 8px;'>Linear Regression with Least Squares implementation featuring analytical solution computation, residual analysis, and statistical inference capabilities. Provides foundation for understanding linear relationships in data.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python 3.7+
- **Core Dependency:** NumPy (for numerical computations)
- **Optional:** Matplotlib (for visualization in examples)

### Installation

Build the ML library from source:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/stericishere/self-built-ml-library
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd self-built-ml-library
    ```

3. **Install the dependencies:**

    ```sh
    ❯ pip install numpy matplotlib
    ```

### Usage

**Basic Neural Network Example:**

```python
from ML.Neural_Network.nn import NeuralNet
from ML.Neural_Network.layers import Linear, Tanh
from ML.Neural_Network.train import train
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

**Run Example Applications:**

```sh
# Train XOR neural network
❯ python ML/Neural\ Network/xor.py

# Train FizzBuzz classifier  
❯ python ML/Neural\ Network/fizzbuzz.py

# Run logistic regression
❯ python ML/logistic\ regression/run_logistic_regression\ copy.py
```

### Testing

Verify gradient implementations:

```sh
❯ python ML/logistic\ regression/check_grad.py
```

Run individual algorithm tests by executing their respective Python files in each module directory.

---

## Roadmap

- [X] **`Core Neural Networks`**: <strike>Feedforward networks with backpropagation</strike>
- [X] **`Basic Algorithms`**: <strike>Decision Trees, KNN, Logistic Regression</strike>
- [X] **`Probability Models`**: <strike>Naive Bayes, Linear Regression</strike>
- [ ] **`Advanced Optimizers`**: Adam, RMSprop, Momentum SGD
- [ ] **`Regularization`**: Dropout, L1/L2 regularization, Batch normalization
- [ ] **`Convolutional Layers`**: CNN implementation for image processing
- [ ] **`Recurrent Networks`**: LSTM/GRU for sequence modeling
- [ ] **`Advanced Loss Functions`**: Cross-entropy, Huber loss, Custom losses
- [ ] **`Model Serialization`**: Save/load trained models
- [ ] **`Visualization Tools`**: Training curves, model architecture plots
- [ ] **`Performance Optimization`**: Vectorization improvements, Memory efficiency

---

## Contributing

Contributions are welcome! This project is designed for educational purposes and learning.

- **💬 [Join the Discussions](https://github.com/stericishere/self-built-ml-library/discussions)**: Share insights, provide feedback, or ask questions about implementations.
- **🐛 [Report Issues](https://github.com/stericishere/self-built-ml-library/issues)**: Submit bugs or request new algorithm implementations.
- **💡 [Submit Pull Requests](https://github.com/stericishere/self-built-ml-library/pulls)**: Contribute new algorithms or improvements.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine.
   ```sh
   git clone https://github.com/your-username/self-built-ml-library
   ```
3. **Create a New Branch**: Always work on a new branch with a descriptive name.
   ```sh
   git checkout -b feature/new-algorithm
   ```
4. **Follow the Code Style**: 
   - Use type annotations
   - Include comprehensive docstrings
   - Follow the existing module structure
   - Implement both forward and backward passes for neural network components
5. **Add Tests**: Include example usage and gradient checking where applicable.
6. **Commit Your Changes**: Write clear commit messages.
   ```sh
   git commit -m 'Add Adam optimizer implementation'
   ```
7. **Push to GitHub**: Push changes to your forked repository.
   ```sh
   git push origin feature/new-algorithm
   ```
8. **Submit a Pull Request**: Create a PR with detailed description of changes.

</details>

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Educational Purpose**: This framework serves as a learning tool for understanding ML fundamentals
- **NumPy Community**: For providing the foundational numerical computing library
- **Deep Learning Pioneers**: For establishing the mathematical foundations implemented here
- **Open Source ML Community**: For inspiration and reference implementations

*This framework represents a journey in understanding machine learning from first principles. Each algorithm is implemented to provide deep insight into the mathematical concepts that power modern AI systems.*

<div align="right">

[![][back-to-top]](#top)

</div>

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square

--- 