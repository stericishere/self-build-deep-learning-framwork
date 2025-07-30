from check_grad import check_grad
from logistic import logistic, logistic_predict, evaluate, sigmoid
import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = None, None
    valid_inputs, valid_targets = None, None

    N, M = train_inputs.shape


    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.00, # L2 regularization
        "num_iterations": 1000,
    }
    weights = np.random.randn(M + 1, 1) * np.sqrt(1.0 / M) 
    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    val_losses = []
    train_losses = []
    for i in range(hyperparameters["num_iterations"]):
        train_loss, gradient, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = weights - hyperparameters["learning_rate"] * gradient
        y_val = logistic_predict(weights, valid_inputs)
        val_loss = -np.mean(valid_targets * np.log(y_val + 1e-8) + (1-valid_targets) * np.log(1-y_val + 1e-8))
        val_losses.append(val_loss)
        train_losses.append(train_loss)
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {train_loss}")
            print(f"Val Acc: {np.mean((y_val > 0.5) == valid_targets):.2f}")

    test_inputs, test_targets = None, None
    predictions = logistic_predict(weights, test_inputs)
    accuracy  = np.mean((predictions > 0.5) == test_targets)
    print(f"Accuracy: {accuracy}")
    iterations = np.arange(0, hyperparameters["num_iterations"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, 'orange', label='Training Loss', linewidth=2)
    plt.plot(iterations, val_losses, 'blue', label='Validation Loss', linewidth=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training vs Validation Loss for mnist_train_small')
    plt.savefig('combined_loss_curves_mnist_train_small.png')
    plt.show()


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic, weights, 0.001, data, targets, hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
