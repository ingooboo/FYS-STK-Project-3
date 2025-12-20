"""Activation functions and derivative helpers."""

import autograd.numpy as np

# Identity activation: f(z) = z.
def identity(z):
    """Return input unchanged.

    Args:
        z: Input array or scalar.

    Returns:
        The same value as `z`.
    """
    return z

# Hyperbolic tangent via autograd numpy.
def tanh(z):
    """Compute tanh activation.

    Args:
        z: Input array or scalar.

    Returns:
        Tanh of `z`.
    """
    return np.tanh(z)

def act_tanh(z):
    """Compute tanh manually without np.tanh.

    Args:
        z: Input array or scalar.

    Returns:
        Tanh of `z`.
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

# Logistic sigmoid activation.
def sigmoid(z):
    """Compute logistic sigmoid activation.

    Args:
        z: Input array or scalar.

    Returns:
        Sigmoid of `z`.
    """
    return 1/(1 + np.exp(-z))

# Rectified linear unit.
def relu(z):
    """Compute ReLU activation.

    Args:
        z: Input array or scalar.

    Returns:
        ReLU of `z`.
    """
    return np.maximum(0, z)

# Leaky ReLU with default negative slope.
def lrelu(z, alpha=0.01):
    """Compute leaky ReLU activation.

    Args:
        z: Input array or scalar.
        alpha: Negative-slope coefficient.

    Returns:
        Leaky ReLU of `z`.
    """
    return np.where(z > 0, z, alpha * z)

# Dispatcher that returns the derivative function for a given activation.
def derivate(func: callable):
    """Return derivative function matching the given activation.

    Args:
        func: Activation function to differentiate by name.

    Returns:
        Callable that computes the derivative with respect to its input.
    """
    if func.__name__ == "identity":

        def der_func(X):
            """Derivative of identity.

            Args:
                X: Input array.

            Returns:
                Array of ones with the same shape as `X`.
            """
            return np.ones_like(X)

        return der_func

    if func.__name__ == "relu":

        def der_func(X):
            """Derivative of ReLU.

            Args:
                X: Input array.

            Returns:
                1 where X > 0, else 0.
            """
            return np.where(X > 0, 1, 0)

        return der_func

    elif func.__name__ == "lrelu":

        def der_func(X):
            """Derivative of leaky ReLU with default slope.

            Args:
                X: Input array.

            Returns:
                1 where X > 0, else alpha.
            """
            alpha = 0.01
            return np.where(X > 0, 1, alpha)

        return der_func

    elif func.__name__ == "sigmoid":

        def der_func(X):
            """Derivative of sigmoid.

            Args:
                X: Input array.

            Returns:
                Sigmoid derivative evaluated at `X`.
            """
            s = sigmoid(X)
            return s * (1 - s)
        return der_func  
    
    elif func.__name__ == "tanh":

        def der_func(X):
            """Derivative of tanh.

            Args:
                X: Input array.

            Returns:
                Tanh derivative evaluated at `X`.
            """
            return 1 - np.tanh(X)**2
        return der_func
