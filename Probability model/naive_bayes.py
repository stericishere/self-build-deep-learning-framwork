"""
Question 2 Naive Bayes

Here you should implement and train the Naive Bayes Classifier.
NOTE: Do not modify or add any more import statements.
"""

import numpy as np
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image

def train_mle_estimator(train_images, train_labels):
    """Inputs: train_images, train_labels
    Returns the MLE estimators theta_mle and pi_mle"""

    theta_mle = np.zeros((784, 10))
    pi_mle = np.zeros(10)
    for c in range(10):
        num_of_class_c = np.sum(train_labels[:, c] == 1)
        class_c_images = train_images[train_labels[:, c] == 1, :]
        for j in range(784):
        # x is a row-vector
        # t is a row-vector
        # np.sum(train_labels[:, c] == 1) means the sum of i-th class
        # train_images[train_labels[:, i] == 1] means the xj of the i-th class
        # axis=0 means the sum of the column
            theta_mle[j, c] = np.sum(class_c_images[:, j], axis=0) / num_of_class_c
        pi_mle[c] = num_of_class_c / train_images.shape[0]
    pi_mle = pi_mle / np.sum(pi_mle)
    return theta_mle, pi_mle


def train_map_estimator(train_images, train_labels):
    """Inputs: train_images, train_labels
    Returns the MAP estimators theta_map and pi_map"""
    theta_map = np.zeros((784, 10))
    pi_map = np.zeros(10)
    for c in range(10):
        num_of_class_c = np.sum(train_labels[:, c] == 1)
        class_c_images = train_images[train_labels[:, c] == 1, :]
        for j in range(784):
            theta_map[j, c] = (np.sum(class_c_images[:, j], axis=0) + 2) / (num_of_class_c + 4)
        pi_map[c] = num_of_class_c / train_images.shape[0]
    pi_map = pi_map / np.sum(pi_map)
    return theta_map, pi_map


def log_likelihood(images, theta, pi):
    """Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    log_like = np.zeros((images.shape[0], 10))
    for i in range(images.shape[0]):
        for c in range(10):
            log_prior = np.log(pi[c])
            # Log likelihood
            # theta[:, c] gives us the 784 pixel probabilities for class c
            # CORRECTED: theta is indexed as theta[j, c] not theta[i, c]
            log_likelihood = np.sum(
                images[i, :] * np.log(theta[:, c]+1e-10) + 
                (1 - images[i, :]+1e-10) * np.log(1 - theta[:, c]+1e-10)
            )
            
            # This gives us log p(x^(i), c | theta, pi)
            log_like[i, c] = log_likelihood + log_prior
    # Normalize to get conditional probabilities p(c | x^(i))
    # For each image, normalize across all classes
    # log p(c | x^(i)) = log p(x^(i), c | theta, pi) - log p(x^(i) | theta, pi)
    # log p(x^(i) | theta, pi) = sum_c log p(x, c | theta, pi)   
    
    # We use the log-sum-exp trick to find log p(x^(i))
    # log p(x^(i) | theta, pi) = max(log_like) + log sum_i exp(log_like_i - log_like_max)
    for i in range(images.shape[0]):
        # Use log-sum-exp trick for numerical stability
        max_log_like = np.max(log_like[i, :])
        # find log p(x^(i) | theta, pi)
        log_norm_constant = max_log_like + np.log(np.sum(np.exp(log_like[i, :] - max_log_like)))
        
        # log p(c | x^(i), theta, pi) = log p(x^(i), c | theta, pi) - log p(x^(i) | theta, pi)
        log_like[i, :] = log_like[i, :] - log_norm_constant
    
    return log_like


def predict(log_like):
    """Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    # np.argmax(log_like, axis=1) returns the index of the class with the highest log likelihood for each image
    Max_log_like_each_image = np.argmax(log_like, axis=1)
    
    # Convert to one-hot encoding (same format as the labels)
    num_images = log_like.shape[0]
    num_classes = log_like.shape[1]
    predictions = np.zeros((num_images, num_classes))
    
    for i in range(num_images):
        predictions[i, Max_log_like_each_image[i]] = 1
        
    return predictions


def accuracy(log_like, labels):
    """Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""
    predictions = predict(log_like)
    # labels is one-hot encoding, predictions is one-hot encoding
    # therefore, only 1*1 = 1 (meaning it's correct prediction), others are 0
    correct_predictions = np.sum(predictions * labels)
    total_predictions = labels.shape[0]
    
    acc = correct_predictions / total_predictions
    
    return acc


def main():
    N_data, train_images, train_labels, test_images, test_labels = None, None, None, None, None

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)


if __name__ == "__main__":
    main()
