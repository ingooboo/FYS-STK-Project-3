"""Cost functions for the 1D diffusion equation PINN."""

##########################################################################################################
# IMPORT NECESSARY PACKAGES :
##########################################################################################################
from autograd import jacobian,hessian,grad
from src.trial_func import trial_function_autograd, trial_function_autograd_C, trial_function_pyTorch
import torch
# Device, this will use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################################
# COST FUNCTION :
##########################################################################################################
def cost_function_autograd(deep_params, 
                           x,
                           t, 
                           deep_neural_network, 
                           activation_function,
                           debug=False):
    """Compute PDE residual cost using autograd and a function model.

    Args:
        deep_params: List of weight matrices with bias rows.
        x: Spatial grid.
        t: Time grid.
        deep_neural_network: Network callable.
        activation_function: Activation function.
        debug: If True, print debug information.

    Returns:
        Average squared residual cost over the grid.
    """
    # Initialize cost sum
    cost_sum = 0
    # Store results for debugging if needed
    res = []
    # Define jacobian and hessian functions, to compute derivatives
    # Here 1 and 0 refer to the position of t and x in the trial function arguments
    # 0 = x, 1 = t
    psi_t_jacobian_func = jacobian(trial_function_autograd, 1)
    psi_t_hessian_func  = hessian(trial_function_autograd, 0)
    # Loop over all x and t values to compute the cost
    for x_ in x:
        for t_ in t: 
            # Compute derivatives
            psi_t_jacobian = psi_t_jacobian_func(x_, t_, deep_neural_network, deep_params, activation_function, debug)
            psi_t_hessian  = psi_t_hessian_func(x_,  t_, deep_neural_network, deep_params, activation_function, debug)
            psi_t_dt  = psi_t_jacobian
            psi_t_d2x = psi_t_hessian
            # Compute squared error for the PDE residual
            err_sqr = ((psi_t_dt - psi_t_d2x))**2 
            # Accumulate cost
            cost_sum += err_sqr
            res.append(err_sqr)
    # Compute average cost
    cost_avg = cost_sum / (len(x)*len(t))
    # Return average cost         
    return cost_avg

def cost_function_autograd_C(deep_params,
                             x,
                             t,
                             model,
                             debug=False):
    """Compute PDE residual cost using autograd and a model object.

    Args:
        deep_params: List of weight matrices with bias rows.
        x: Spatial grid.
        t: Time grid.
        model: Model object supporting set_params and call.
        debug: If True, print debug information.

    Returns:
        Average squared residual cost over the grid.
    """
    # Initialize cost sum
    cost_sum = 0
    # Store results for debugging if needed
    res = []
    # Define jacobian and hessian functions, to compute derivatives
    # Here 1 and 0 refer to the position of t and x in the trial function arguments
    # 0 = x, 1 = t
    psi_t_jacobian_func = jacobian(trial_function_autograd_C, 1)
    psi_t_hessian_func  = hessian(trial_function_autograd_C, 0)
    # Loop over all x and t values to compute the cost
    for x_ in x:
        for t_ in t: 
            # Compute derivatives
            psi_t_jacobian = psi_t_jacobian_func(x_, t_, deep_params, model, debug)
            psi_t_hessian  = psi_t_hessian_func(x_,  t_, deep_params, model, debug)
            psi_t_dt  = psi_t_jacobian
            psi_t_d2x = psi_t_hessian
            # Compute squared error for the PDE residual
            err_sqr = ((psi_t_dt - psi_t_d2x))**2 
            # Accumulate cost
            cost_sum += err_sqr
            res.append(err_sqr)
    # Compute average cost
    cost_avg = cost_sum / (len(x)*len(t))
    # Return average cost         
    return cost_avg

def cost_function_pyTorch(x,
                          t, 
                          model,
                          debug=False):
    """Compute PDE residual cost using PyTorch autograd.

    Args:
        x: Spatial tensor (1D).
        t: Time tensor (1D).
        model: PyTorch model.
        debug: If True, print debug information.

    Returns:
        Scalar tensor with average squared residual cost.
    """
    # Make sure x,t are tensors on device
    x = x.to(device)
    t = t.to(device)
    # Create meshgrid then flatten
    T, X = torch.meshgrid(t, x, indexing='ij')  # shape (len(t), len(x))
    T_flat = T.reshape(-1)
    X_flat = X.reshape(-1)
    # IMPORTANT: these must require grad and be the tensors used to compute g_trial
    T_flat = T_flat.reshape(-1).clone().detach().to(device).requires_grad_(True)
    X_flat = X_flat.reshape(-1).clone().detach().to(device).requires_grad_(True)
    psi_trial = trial_function_pyTorch(X_flat, T_flat, model, debug)
    # First derivative w.r.t. time for each point (batch)
    ones = torch.ones_like(psi_trial, device=device)
    psi_t = torch.autograd.grad(
        outputs=psi_trial,
        inputs=T_flat,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # First derivative w.r.t. space (x) for each point (batch)
    psi_x = torch.autograd.grad(
        outputs=psi_trial,
        inputs=X_flat,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # second derivative w.r.t. space (x) for each point (batch)
    psi_xx = torch.autograd.grad(
        outputs=psi_x,
        inputs=X_flat,
        grad_outputs=torch.ones_like(psi_x, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # PDE residual for e.g. psi_t - psi_xx 
    res = psi_t - psi_xx
    # Squared residuals
    cost = (res**2)
    # Average cost over all points
    cost_avg = cost.mean()
    # Return average cost
    return cost_avg

##########################################################################################################
# GRADIENT OF COST FUNCTION :
##########################################################################################################
def gradient_cost_function(P, x, t, deep_neural_network, activation_function):
    """Return gradient of the autograd cost function.

    Args:
        P: Model parameters.
        x: Spatial grid.
        t: Time grid.
        deep_neural_network: Network callable.
        activation_function: Activation function.

    Returns:
        Gradient of the cost with respect to parameters.
    """
    cost_function_grad = grad(cost_function_autograd,0)
    return cost_function_grad((P, x, t, deep_neural_network, activation_function))

def g_trial_batch_cost(model, x_batch, t_batch):
    """Compute mean squared trial-function value for a batch.

    Args:
        model: PyTorch model.
        x_batch: Spatial batch tensor.
        t_batch: Time batch tensor.

    Returns:
        Mean squared value of trial function on the batch.
    """
    # x_batch and t_batch are 1D tensors of same length (batch_size)
    R = trial_function_pyTorch(x_batch, t_batch, model)   # note order (t,x,model)
    return (R**2).mean()
