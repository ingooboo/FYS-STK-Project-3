##########################################################################################################
# TRIAL FUNCTION FOR THE ONE-DIMENSIONAL DIFFUSION EQUATION :
##########################################################################################################
# Project 3 - FYS-STK4155 :
# Authors : Ingvild Olden and Jenny Guldvog 

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
    '''
    Calculating the initial conditions for the trial funciton, using autograd.numpy implementation.
    
    Parameters:
    --------
    x :
        The input of x, as a value in the list of discretized x-values.
    debug : bool
        If True, print debug information.
    
    Returns:
    --------
        The calculated initial condition at t=0.
    '''
    u_t0 = np.sin(np.pi*x)
    if debug:
        print("Initial condition at t=0 (autograd): ")
        print(u_t0.shape)
        print(u_t0)
    return u_t0

def u_t0_pyTorch(x,
                 debug=False):
    '''
    Calculating the initial conditions for the trial funciton, using pyTorch tensors implementation.
    
    Parameters:
    --------
    x :
        The input of x, as a value in the torch-list of discretized x-values.
    debug : bool
        If True, print debug information.
    
    Returns: 
    --------
        The calculated initial condition at t=0.
    '''
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
    '''
    Trial function for the one-dimensional diffusion equation, using autograd implementation.
    
    Parameters:
    --------
    x :
        The input of x, as a value in the list of discretized x-values.
    t : 
        The input of t, as a value in the list of discretized t-values.
    deep_neural_network : 
        The neural network used in the trial function.
    deep_params : 
        List containing the weights and biases for each layer. 
        Each element in the list is a 2D array where each row corresponds 
        to the weights and bias for one neuron in that layer.
    activation_function : 
        The activation function used in the neural network
    debug : bool
        If True, print debug information.
    
    Returns:
    --------
        The calculated trial function for given x and t
    '''
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
    '''
    Trial function for the one-dimensional diffusion equation, using autograd implementation.
    
    Parameters:
    --------
    deep_params : 
        List containing the weights and biases for each layer. 
        Each element in the list is a 2D array where each row corresponds 
        to the weights and bias for one neuron in that layer.
    x :
        The input of x, as a value in the list of discretized x-values.
    t : 
        The input of t, as a value in the list of discretized t-values.
    deep_neural_network : 
        The neural network used in the trial function.
    deep_params : 
        List containing the weights and biases for each layer. 
        Each element in the list is a 2D array where each row corresponds 
        to the weights and bias for one neuron in that layer.
    activation_function : 
        The activation function used in the neural network
    debug : bool
        If True, print debug information.
    
    Returns:
    --------
        The calculated trial function for given x and t
    '''
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
    '''
    Trial function for the one-dimensional diffusion equation, using pyTorch implementation.
    
    Parameters:
    --------
    x :
        The input of x, as a value in the torch-list of discretized x-values.
        Tensor that already have requires_grad=True and are on device. Do not clone().detach() here.
    t : 
        The input of t, as a value in the torch-list of discretized x-values.
        Tensor that already have requires_grad=True and are on device. Do not clone().detach() here.
    model : 
        The neural network used in the trial function. 
        Here pyTorch is implemented, and the activation function is chosen before this step.
    debug : bool
        If True, print debug information.

    Returns:
    --------
        The calculated trial function for given x and t
    '''
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