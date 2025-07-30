from Array import Array
import numpy as np


def logistic_predict(weights: Array, data: Array) -> Array:
    """Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    bias = np.ones((data.shape[0], 1))
    data_with_bias = np.concatenate((data, bias), axis=1)
    z = np.dot(data_with_bias, weights) # Nx1
    predictions = sigmoid(z)
    # have to use data_with_bias.T because data_with_bias is Nx(M+1) and train_targets is Nx1
    # 1/N * X^T * (error = y^ - Y)

    y = predictions
    return y


def evaluate(targets: Array, y: Array) -> tuple[float, float]:
    """Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """

    ce = -np.mean(targets * np.log(y + 1e-8) + (1-targets) * np.log(1-y + 1e-8))
    frac_correct = np.mean((y>0.5) == targets)

    return ce, frac_correct


def logistic(
    weights: Array, data: Array, targets: Array, hyperparameters: dict
) -> tuple[float, Array, Array]:
    """Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    average_loss = -np.mean(targets * np.log(y + 1e-8) + (1-targets) * np.log(1-y + 1e-8))
    l2_penalty = 0.5 * hyperparameters["weight_regularization"]  * np.sum(weights[:-1] ** 2)  # Don't regularize bias (weight[0])
    total_loss = average_loss + l2_penalty
    bias = np.ones((data.shape[0], 1))
    data_with_bias = np.concatenate((data, bias), axis=1)
    # cross entropy gradient
    cross_entropy_gradient = 1/data_with_bias.shape[0] * np.dot(data_with_bias.T, y - targets)
    # l2 penalty gradient
    l2_penalty_gradient = np.zeros_like(weights)  # Same shape as weights: (M+1) × 1
    l2_penalty_gradient[:-1] = hyperparameters["weight_regularization"] * weights[:-1]
    

    f = total_loss
    # total gradient
    df = cross_entropy_gradient + l2_penalty_gradient.reshape(-1, 1)
    return f, df, y

def sigmoid(x: Array) -> Array:
    return 1 / (1 + np.exp(-x))