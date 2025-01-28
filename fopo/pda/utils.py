import numpy as np 
from typing import Any, Optional




def softmax(x):
    # Subtract the maximum value in the input array for numerical stability
    x = x - np.max(x)

    # Compute the exponentials of the input array
    exp_x = np.exp(x)

    # Compute the softmax values by dividing the exponentials by their sum
    softmax_x = exp_x / np.sum(exp_x)

    return softmax_x


def random_stump_feature(input_dim: int, num_features: int, seed: Optional[Any] = None):
    """
    Create a random stump feature map function.

    Parameters:
    input_dim: int
        The dimension of the input data.
    num_features: int
        The number of random stump features to generate.
    seed: int, optional
        The random seed for reproducibility.

    Returns:
    mapper: function
        A function that takes a data point as input and returns a feature map using random stump features.
    """
    if seed is not None:
        np.random.seed(seed)
    # make sure the input of the feature is within high and low
    thresholds = np.random.uniform(low=-5, high=-5, size=num_features)
    dimensions = np.random.randint(0, input_dim, size=num_features)

    def mapper(*x):
        """
        Compute the random stump feature map for the input data point x.

        Parameters:
        x: a vector of numpy array

        Returns:
        feature_map: numpy array, shape (num_features,)
            The random stump feature map of the input data point.
        """
        # first vectorize x
        vx = np.concatenate([np.ravel(e) for e in x])
        return (vx[dimensions] > thresholds).astype(int) * np.sqrt(2.0 / num_features)

    return mapper


def random_fourier_feature(input_dim: int, num_features: int, gamma: float=1.0, seed: Optional[Any] = None):
    """
    Generate a function that computes the Random Fourier Features for input data.

    Parameters:
    input_dim: int
        The dimension of the input data.
    num_features: int
        The number of random Fourier features to generate.
    gamma: float, optional, default: 1.0
        The gamma parameter for the Gaussian (RBF) kernel.
    seed: int, optional
        The random seed for reproducibility.

    Returns:
    mapper: function
        A function that computes the Random Fourier Features for input data.
    """

    if seed is not None:
        np.random.seed(seed)

    W = np.random.normal(0, np.sqrt(2 * gamma), (input_dim, num_features))
    b = np.random.uniform(-np.pi, np.pi, num_features)

    def mapper(*x):
        """
        Compute the Random Fourier Features for input data x (could contain multiple elements (e.g. state, action)).

        Parameters:
        x: a vector of numpy array

        Returns:
        z: numpy array, shape (num_features)
            The transformed data with Random Fourier Features.
        """
        # first vectorize x
        vx = np.concatenate([np.ravel(e) for e in x])
        z = np.dot(W.T, vx) + b
        z = np.sqrt(2.0 / num_features) * np.cos(z)
        return z

    return mapper


def bisection_search(f, a, b, tol=1e-6, max_iters=100):
    """
    Finds a root of the function f(x) on the interval [a, b] using the bisection search algorithm.
    
    Parameters:
        f (callable): A function of one variable.
        a, b (float): The endpoints of the interval to search for a root.
        tol (float, optional): The desired tolerance for the root. Default is 1e-6.
        max_iters (int, optional): The maximum number of iterations allowed. Default is 100.
    
    Returns:
        float: The approximate root of f(x) on the interval [a, b].
    """
    if abs(f(a)) <= tol:
        return a
    if abs(f(a)) <= tol:
        return b
    if f(a) * f(b) >= tol:
        raise ValueError("The function does not change sign on the interval.")
    
    x = (a + b) / 2
    iters = 0
    
    while abs(f(x)) > tol and iters < max_iters:
        if f(x) == 0:
            return x
        elif f(x) * f(a) < 0:
            b = x
        else:
            a = x
        x = (a + b) / 2
        iters += 1
    
    if abs(f(x)) > tol:
        raise RuntimeError("Bisection search did not converge within the specified number of iterations.")
    
    return x



def prox_update_Tsallis_state(pi: np.array, G: np.array, eta: float, index: float, tol: float) -> np.array:
        """
        solve Tsallis policy update for a given state
        \min_{p} \eta [<G(s), p>] + D^{p}_{\pi(s)}, where divergence is given by Tsallis entropy
        this update is vectorized
        pi: policy 
        G: Q-function
        index: entropic index
        eta: stepsize 
        epsilon: target precision for solving the proximal update
        Details can be found at: https://arxiv.org/abs/2303.04386, Prop 5.2
        """
        q = eta * G + index * np.power(pi, index-1)
        phi = lambda u: np.sum(np.power(index/(q - u), 1/(1-index))) - 1.
        l = np.min(q) - index * np.power(len(q), 1-index)
        h = np.min(q) - index 
        u = bisection_search(phi, l, h, tol)
        p = np.power(index/(q - u), 1/(1-index))
        p /= np.sum(p)
        return p 