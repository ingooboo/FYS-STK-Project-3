"""Helpers to compute PINN and analytical solution grids."""
import autograd.numpy as np
from .analytical_func import g_analytic
from .trial_func import trial_function_autograd, trial_function_pyTorch
from .neural_network_pyTorch import Net, device

def PINN_solution(Nx,Nt,x,t,model_imp,P,num_hidden_neurons,activation_func,autograd_or_pytorch):
    """Compute PINN solution grid using autograd or PyTorch backend.

    Args:
        Nx: Number of spatial points.
        Nt: Number of time points.
        x: Spatial grid.
        t: Time grid.
        model_imp: Model callable or class.
        P: Model parameters or state dict.
        num_hidden_neurons: Hidden layer widths.
        activation_func: Activation function name/callable.
        autograd_or_pytorch: Backend selector ('autograd' or 'pytorch').

    Returns:
        2D array with solution values on the (x, t) grid.
    """
    temp_u = np.zeros((Nx, Nt))
    # Evaluate solution at each grid point.
    for i, x_ in enumerate(x):
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            if autograd_or_pytorch == 'pytorch':
                # Build a PyTorch model, load weights, and evaluate trial function.
                model = model_imp(network=num_hidden_neurons, activation=activation_func).to(device)
                model.load_state_dict(P)
                temp_u[i,j] = trial_function_pyTorch(x_, t_, model).detach().cpu().numpy()
            if autograd_or_pytorch == 'autograd':
                # Call autograd-based trial function.
                temp_u[i,j] = trial_function_autograd(x_, t_, model_imp, P, activation_function=activation_func)
    return temp_u

def analytical_solution(Nx,Nt,x,t):
    """Compute analytical solution grid.

    Args:
        Nx: Number of spatial points.
        Nt: Number of time points.
        x: Spatial grid.
        t: Time grid.

    Returns:
        2D array with analytical solution values on the (x, t) grid.
    """
    analytical_temp = np.zeros((Nx, Nt))
    # Evaluate analytical solution at each grid point.
    for i, x_ in enumerate(x):
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            analytical_temp[i,j] = g_analytic(point)
    return analytical_temp
