"""Training utilities for the autograd-based PINN."""

##########################################################################################################
# IMPORT NECESSARY PACKAGES :
##########################################################################################################
import numpy.random as npr
import autograd.numpy as np
from math import ceil
from autograd import grad
from .cost_func import cost_function_autograd_C as cost_function
from .neural_network import DeepNeuralNetwork

##########################################################################################################
# ADAM OPTIMIZER :
##########################################################################################################
def adam_update(params, 
                grads, 
                m, 
                v, 
                t, 
                learning_rate=0.001,
                beta1=0.9, 
                beta2=0.999, 
                epsilon=1e-8):  
    """Perform Adam optimization update on parameter lists.

    Args:
        params: Current parameters.
        grads: Gradients for each parameter.
        m: First-moment estimates.
        v: Second-moment estimates.
        t: Time step for bias correction.
        learning_rate: Step size.
        beta1: Decay rate for first moment.
        beta2: Decay rate for second moment.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Tuple of (updated_params, new_m, new_v).
    """
    # Initialize lists to hold updated parameters and moment estimates
    updated_params = []
    new_m = []
    new_v = []
    # Update each parameter using Adam optimization
    for p, g, m_i, v_i in zip(params, grads, m, v):
        m_i = beta1 * m_i + (1 - beta1) * g
        v_i = beta2 * v_i + (1 - beta2) * (g ** 2)
        m_hat = m_i / (1 - beta1 ** t)
        v_hat = v_i / (1 - beta2 ** t)
        p = p - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        updated_params.append(p)
        new_m.append(m_i)
        new_v.append(v_i)
    return updated_params, new_m, new_v

##########################################################################################################
# TRAIN THE DEEP NEURAL NETWORK TO SOLVE THE PDE :
##########################################################################################################
def train_PINN_autograd(x,
                        t,
                        num_neurons,
                        epochs,
                        learning_rate,
                        activation_function,
                        seed,
                        optimization_method, # 'GD', 'GD-adam', 'SGD-adam'
                        batch_size=None,
                        shuffle=True,
                        replacement=False,
                        verbose=False,
                        debug=False):
    """Train the autograd PINN with GD/Adam/SGD-Adam.

    Args:
        x: 1D spatial grid.
        t: 1D time grid.
        num_neurons: Hidden layer widths.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        activation_function: Activation function name or callable.
        seed: RNG seed.
        optimization_method: 'GD', 'GD-adam', or 'SGD-adam'.
        batch_size: Mini-batch size for SGD-Adam.
        shuffle: If True, shuffle batches each epoch.
        replacement: If True, sample batches with replacement.
        verbose: If True, print training progress.
        debug: If True, print debug information from model/cost.

    Returns:
        Tuple of (trained_params, history).
    """
    # Set RNG seed for reproducible initialization
    npr.seed(seed)
    rng = npr.RandomState(seed)
    if optimization_method == 'SGD-adam':
        # Build full meshgrid coordinates (flattened) once (match PyTorch behavior)
        # Use indexing='ij' to match torch.meshgrid(..., indexing='ij')
        T_full, X_full = np.meshgrid(t, x, indexing='ij')
        T_flat = T_full.ravel()
        X_flat = X_full.ravel()
        n_samples = T_flat.shape[0]
    # Set up initial weigths and biases
    N_hidden = len(num_neurons)
    # Initialize the list of parameters:
    P = [None]*(N_hidden + 1) # + 1 to include the output layer
    P[0] = npr.randn(num_neurons[0], 2 + 1 ) # 2 since we have two points, +1 to include bias
    for l in range(1,N_hidden):
        P[l] = npr.randn(num_neurons[l], num_neurons[l-1] + 1) # +1 to include bias
    # For the output layer
    P[-1] = npr.randn(1, num_neurons[-1] + 1 ) # +1 since bias is included
    # Call the model : 
    model = DeepNeuralNetwork(P, activation_function)
    # --- begin: added history collection ---
    history = {'epoch': [], 'cost': []}
    init_cost = cost_function(P, x, t, model, debug)
    history['epoch'].append(0)
    history['cost'].append(init_cost)
    # --- end: added history collection ---
    print('Initial cost: ',cost_function(P, x, t, model, debug))
    # gradient of cost_function w.r.t. parameters P (index 0)
    cost_function_grad = grad(cost_function, 0)
    if optimization_method != 'GD':
        # Initialize Adam momentum parameters once
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
    if optimization_method == 'GD' or optimization_method == 'GD-adam':
        # Let the update be done epoch times
        for i in range(epochs):
            cost_grad =  cost_function_grad(P, x, t, model, debug)
            if optimization_method == 'GD':
                for l in range(N_hidden+1):
                    P[l] = P[l] - learning_rate * cost_grad[l]
            elif optimization_method == 'GD-adam':
                # Update parameters using Adam optimizer
                P, m, v = adam_update(P, cost_grad, m, v, t=i+1, learning_rate=learning_rate)
            # --- begin: add per-epoch history entry (after the update) ---
            curr_cost = cost_function(P, x, t, model, debug)
            if verbose:
                print(f'Epoch {i+1}, Cost: {curr_cost}')
            history['epoch'].append(i+1)
            history['cost'].append(curr_cost)
            # --- end: add per-epoch history entry ---
    iter_count = 0
    #epoch = 0
    # Helper: yield minibatch index arrays (no replacement) for one epoch
    def epoch_minibatch_indices_no_replacement(order, batch_size):
        # ceil is used to ensure all samples are used
        # ceil(n_samples / batch_size) gives number of batches needed
        # ceil is a function that returns the smallest integer >= input
        n_batches = ceil(n_samples / batch_size)
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)
            yield order[start:end]
    if optimization_method == 'SGD-adam':
        # For replacement mode choose similar number of batches per epoch
        batches_per_epoch_replacement = int(ceil(n_samples / float(batch_size)))
    if optimization_method == 'SGD-adam':
        for i in range(1, epochs+1):
            if replacement:
                # sample each minibatch with replacement for one epoch
                for _ in range(batches_per_epoch_replacement):
                    idx = rng.choice(n_samples, batch_size, replace=True)
                    xb, tb = X_flat[idx], T_flat[idx]
                    grads = cost_function_grad(P, xb, tb, model, debug)
                    iter_count += 1
                    P, m, v = adam_update(P, grads, m, v, t=iter_count, learning_rate=learning_rate)
            else:
                # no replacement: shuffle or fixed order
                if shuffle:
                    order = rng.permutation(n_samples)
                else:
                    order = np.arange(n_samples)
                for idx in epoch_minibatch_indices_no_replacement(order, batch_size):
                    xb, tb = X_flat[idx], T_flat[idx]
                    grads = cost_function_grad(P, xb, tb, model, debug)
                    iter_count += 1
                    P, m, v = adam_update(P, grads, m, v, t=iter_count, learning_rate=learning_rate)
            # --- begin: add per-epoch history entry (after the update) ---
            curr_cost = cost_function(P, x, t, model, debug)
            if verbose:
                print(f'Epoch {i} Iter {iter_count} Cost: {curr_cost}')
            history['epoch'].append(i)
            history['cost'].append(curr_cost)
            # --- end: add per-epoch history entry ---
    print('Final cost: ',cost_function(P, x, t, model, debug))
    # Returns the optimized parameters
    return P, history
