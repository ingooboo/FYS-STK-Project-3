import autograd.numpy as np

def identity(z):
    return z

def tanh(z):
    return np.tanh(z)

def act_tanh(z):
    """ Without np. """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def lrelu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# Derivatives of the activation functions
def derivate(func: callable):
    if func.__name__ == "identity":

        def der_func(X):
            return np.ones_like(X)

        return der_func

    if func.__name__ == "relu":

        def der_func(X):
            return np.where(X > 0, 1, 0)

        return der_func

    elif func.__name__ == "lrelu":

        def der_func(X):
            alpha = 0.01
            return np.where(X > 0, 1, alpha)

        return der_func

    elif func.__name__ == "sigmoid":

        def der_func(X):
            s = sigmoid(X)
            return s * (1 - s)
        return der_func  
    
    elif func.__name__ == "tanh":

        def der_func(X):
            return 1 - np.tanh(X)**2
        return der_func