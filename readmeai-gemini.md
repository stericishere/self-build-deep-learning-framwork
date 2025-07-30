<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# SELF-BUILD-DEEP-LEARNING-FRAMWORK

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/stericishere/self-build-deep-learning-framwork?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/stericishere/self-build-deep-learning-framwork?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/stericishere/self-build-deep-learning-framwork?style=default&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/stericishere/self-build-deep-learning-framwork?style=default&color=0080ff" alt="repo-language-count">

<!-- default option, no dependency badges. -->


<!-- default option, no dependency badges. -->

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



---

## Features

<code>❯ REPLACE-ME</code>

---

## Project Structure

```sh
└── self-build-deep-learning-framwork/
    ├── ML
    │   ├── Decision Trees
    │   ├── KNN
    │   ├── Neural Network
    │   ├── logistic regression
    │   └── loss_function.py
    ├── Probability model
    │   ├── LRLS.py
    │   └── naive_bayes.py
    ├── README.md
    ├── __pycache__
    │   ├── data.cpython-311.pyc
    │   ├── layers.cpython-311.pyc
    │   ├── loss_function.cpython-311.pyc
    │   ├── nn.cpython-311.pyc
    │   ├── optim.cpython-311.pyc
    │   ├── tensor.cpython-311.pyc
    │   └── train.cpython-311.pyc
    ├── data.py
    └── tensor.py
```

### Project Index

<details open>
	<summary><b><code>SELF-BUILD-DEEP-LEARNING-FRAMWORK/</code></b></summary>
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
					<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/tensor.py'>tensor.py</a></b></td>
					<td style='padding: 8px;'>- Tensor.py establishes a type alias, defining <code>Tensor</code> as a synonym for NumPys <code>ndarray</code><br>- Within the broader project, this facilitates consistent use of a specific data structure—the NumPy array—for representing tensors, thereby improving code readability and maintainability across the entire application<br>- This promotes uniformity in handling multi-dimensional numerical data.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/data.py'>data.py</a></b></td>
					<td style='padding: 8px;'>- Data.py provides data iteration functionality for the neural network<br>- It defines tools to efficiently process input and target data in batches, enabling mini-batch gradient descent training<br>- A <code>BatchIterator</code> class facilitates shuffling and batching of data, crucial for effective model training and preventing overfitting<br>- The code ensures consistent input shapes for the neural network.</td>
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
					<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/loss_function.py'>loss_function.py</a></b></td>
					<td style='padding: 8px;'>- Loss_function.py` defines a base class and an implementation for calculating Mean Squared Error (MSE), crucial components within a machine learning model<br>- It provides methods to compute the loss value and its gradient, guiding model optimization by quantifying prediction accuracy<br>- This module contributes to the overall projects ability to train and improve its predictive capabilities.</td>
				</tr>
			</table>
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
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/logistic regression/run_logistic_regression copy.py'>run_logistic_regression copy.py</a></b></td>
							<td style='padding: 8px;'>- The script implements logistic regression for binary classification<br>- It trains a model using gradient descent, periodically evaluating performance on training and validation sets<br>- Gradient checking ensures correctness<br>- Finally, it reports test accuracy and visualizes training progress by plotting loss curves, demonstrating model convergence and generalization ability.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/logistic regression/logistic.py'>logistic.py</a></b></td>
							<td style='padding: 8px;'>- The <code>logistic.py</code> module implements a logistic regression model<br>- It computes predictions, evaluates model performance using cross-entropy and classification accuracy, and calculates the cost function and its gradient for weight updates<br>- The module facilitates training and prediction within a broader machine learning project, likely utilizing a gradient-descent based optimization algorithm.</td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/logistic regression/check_grad.py'>check_grad.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
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
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/KNN/run_kn.pyn'>run_kn.pyn</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
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
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Decision Trees/DecisionTree.py'>DecisionTree.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Decision Trees/Random_forest.py'>Random_forest.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
					</table>
				</blockquote>
			</details>
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
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Neural Network/nn.py'>nn.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Neural Network/fizzbuzz.py'>fizzbuzz.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Neural Network/train.py'>train.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Neural Network/xor.py'>xor.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Neural Network/layers.py'>layers.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
						</tr>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/ML/Neural Network/optim.py'>optim.py</a></b></td>
							<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
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
					<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/Probability model/naive_bayes.py'>naive_bayes.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/stericishere/self-build-deep-learning-framwork/blob/master/Probability model/LRLS.py'>LRLS.py</a></b></td>
					<td style='padding: 8px;'>Code>❯ REPLACE-ME</code></td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python

### Installation

Build self-build-deep-learning-framwork from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/stericishere/self-build-deep-learning-framwork
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd self-build-deep-learning-framwork
    ```

3. **Install the dependencies:**

echo 'INSERT-INSTALL-COMMAND-HERE'

### Usage

Run the project with:

echo 'INSERT-RUN-COMMAND-HERE'

### Testing

Self-build-deep-learning-framwork uses the {__test_framework__} test framework. Run the test suite with:

echo 'INSERT-TEST-COMMAND-HERE'

---

## Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/stericishere/self-build-deep-learning-framwork/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/stericishere/self-build-deep-learning-framwork/issues)**: Submit bugs found or log feature requests for the `self-build-deep-learning-framwork` project.
- **💡 [Submit Pull Requests](https://github.com/stericishere/self-build-deep-learning-framwork/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/stericishere/self-build-deep-learning-framwork
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/stericishere/self-build-deep-learning-framwork/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=stericishere/self-build-deep-learning-framwork">
   </a>
</p>
</details>

---

## License

Self-build-deep-learning-framwork is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
