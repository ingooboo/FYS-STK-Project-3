"""Trial functions for the 1D diffusion equation (autograd and PyTorch)."""

##########################################################################################################
# IMPORT NECESSARY PACKAGES :
##########################################################################################################
import autograd.numpy as np
import torch
# Device, this will use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################################
# INITIAL CONDITIONS AT t=0 :
##########################################################################################################
def u_t0_autograd(x, 
                  debug=False):
    """Compute initial condition at t=0 using autograd numpy.

    Args:
        x: Spatial coordinate(s).
        debug: If True, print debug information.

    Returns:
        Initial condition u(x, 0).
    """
    u_t0 = np.sin(np.pi*x)
    if debug:
        print("Initial condition at t=0 (autograd): ")
        print(u_t0.shape)
        print(u_t0)
    return u_t0

def u_t0_pyTorch(x,
                 debug=False):
    """Compute initial condition at t=0 using PyTorch tensors.

    Args:
        x: Spatial coordinate tensor(s).
        debug: If True, print debug information.

    Returns:
        Initial condition u(x, 0) as a tensor.
    """
    u_t0 = torch.sin(torch.pi * x)
    if debug:
        print("Initial condition at t=0 (pyTorch): ")
        print(u_t0.shape)
        print(u_t0)
    return u_t0

##########################################################################################################
# FULL TRIAL FUNCTION :
##########################################################################################################
def trial_function_autograd(x, 
                            t, 
                            deep_neural_network, 
                            deep_params, 
                            activation_function, 
                            debug=False):
    """Compute the trial function using autograd implementation.

    Args:
        x: Spatial coordinate(s).
        t: Time coordinate(s).
        deep_neural_network: Network function.
        deep_params: List of weight matrices with bias rows.
        activation_function: Activation function.
        debug: If True, print debug information.

    Returns:
        Trial function value at (x, t).
    """
    nn_in = np.array([x, t])
    nn_out = deep_neural_network(deep_params, nn_in, activation_function, debug) 
    B = x*(1-x)*t
    trial_function = u_t0_autograd(x, debug) + B*nn_out
    if debug:
        print("Trial function (autograd): ")
        print(trial_function.shape)
        print(trial_function)
    return trial_function

def trial_function_autograd_C(x, 
                              t, 
                              deep_params,
                              model,
                              debug=False):
    """Compute the trial function using a model object with set_params.

    Args:
        x: Spatial coordinate(s).
        t: Time coordinate(s).
        deep_params: List of weight matrices with bias rows.
        model: Model object that supports set_params and call.
        debug: If True, print debug information.

    Returns:
        Trial function value at (x, t).
    """
    nn_in = np.array([x, t])
    model.set_params(deep_params)
    N = model(nn_in, debug)
    B = x*(1-x)*t
    trial_function = u_t0_autograd(x, debug) + B * N
    if debug:
        print("Trial function (autograd): ")
        print(trial_function.shape)
        print(trial_function)
    return trial_function

def trial_function_pyTorch(x,
                           t, 
                           model,
                           debug=False):
    """Compute the trial function using PyTorch implementation.

    Args:
        x: Spatial tensor (requires_grad=True, on device).
        t: Time tensor (requires_grad=True, on device).
        model: PyTorch model (activation set at construction).
        debug: If True, print debug information.

    Returns:
        Trial function value at (x, t) as a tensor.
    """
    N = model(x, t, debug)
    #if N.dim() > 1:
    #    N = N.squeeze(-1)
    B = x * (1.0 - x) * t
    trial_function = u_t0_pyTorch(x, debug) + B * N
    if debug:
        print("Trial function (pyTorch): ")
        print(trial_function.shape)
        print(trial_function)
    return trial_function
