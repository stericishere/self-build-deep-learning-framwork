from __future__ import print_function
from struct import calcsize
import matplotlib.pyplot as plt
import numpy as np

LAM = 1e-5

# For tpye contracts
Array = np.ndarray

# helper function
def l2(A: Array, B: Array) -> Array:
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist

def LRLS(
    test_datum: Array,
    x_train: Array,
    y_train: Array,
    tau: float,
    lam: float = LAM,
) -> Array:
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    dist = l2(test_datum.reshape(1, -1), x_train) # 1 X N
    dist = dist.flatten() # N x 1
    a_i = np.exp(-dist / (2 * tau**2)) # N x 1
    sum_a = np.sum(a_i, keepdims=True) # N x 1
    a_i = a_i / sum_a # N x 1
    A = np.diag(a_i) # N x N
    # solve the linear system
    XtAX = x_train.T @ A @ x_train
    XtAy = x_train.T @ A @ y_train
    W = np.linalg.solve(XtAX + lam * np.eye(x_train.shape[1]), XtAy)
    y_hat = np.dot(test_datum, W)
    if isinstance(y_hat, np.ndarray):
        y_hat = y_hat.item()  # Convert to scalar
    
    return y_hat

def run_validation(
    x: Array, y: Array, taus: Array, val_frac: float
) -> tuple[list[float], list[float]]:
    """
    Input: x is the N x d design matrix
           y is the 1-dimensional vector of size N
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """
    train_losses = []
    validation_losses = []
    N = x.shape[0]
    train_til_N = int(N * (1 - val_frac))
    train_x = x[:train_til_N,:]
    train_y = y[:train_til_N]
    val_x = x[train_til_N:,:]
    val_y = y[train_til_N:]

    train_size = train_x.shape[0]
    val_size = val_x.shape[0]
    for tau_idx, tau in enumerate(taus):
        print(f"Processing tau {tau_idx+1}/{len(taus)}: {tau:.1f}")
        train_loss = 0
        for i in range(train_x.shape[0]):
            y_pred = LRLS(train_x[i], train_x, train_y, tau)
            train_loss += (y_pred - train_y[i])**2
        train_loss /= train_size # Average training loss
        
        # Compute validation loss (average over validation set)
        val_loss = 0
        for i in range(val_x.shape[0]):
            y_pred = LRLS(val_x[i], train_x, train_y, tau)
            val_loss += (y_pred - val_y[i])**2
        val_loss /= val_size  # Average validation loss
        train_losses.append(train_loss)
        validation_losses.append(val_loss)
        print(f"  -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, validation_losses


if __name__ == "__main__":
    import os

    NUM_TAUS = os.environ.get("NUM_TAUS", 200)

    from sklearn.datasets import fetch_california_housing

    np.random.seed(0)
    # load boston housing prices dataset
    housing = fetch_california_housing()
    n_samples = 500
    x = housing["data"][:n_samples]
    N = x.shape[0]
    # add constant one feature - no bias needed
    x = np.concatenate((np.ones((N, 1)), x), axis=1)
    d = x.shape[1]
    y = housing["target"][:N]
    taus = np.logspace(1, 3, NUM_TAUS)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(taus, train_losses, label="Training Loss")
    plt.semilogx(taus, test_losses, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Tau values (log scale)")
    plt.ylabel("Average squared error loss")
    plt.title("Training and Validation Loss w.r.t Tau")
