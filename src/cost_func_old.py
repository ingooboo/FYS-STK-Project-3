from autograd import jacobian,hessian,grad
from src.trial_func import g_trial, g_trial_pyTorch

def cost_function(P, x, t, deep_neural_network, activation_function):
    cost_sum = 0
    res = []
    g_t_jacobian_func = jacobian(g_trial, 1)
    g_t_hessian_func  = hessian(g_trial, 0)
    for x_ in x:
        for t_ in t:
            g_t_jacobian = g_t_jacobian_func(x_, t_, deep_neural_network, P, activation_function)
            g_t_hessian  = g_t_hessian_func(x_, t_, deep_neural_network, P, activation_function)
            g_t_dt  = g_t_jacobian
            g_t_d2x = g_t_hessian
            err_sqr = ( (g_t_dt - g_t_d2x))**2
            cost_sum += err_sqr
            res.append(err_sqr)
    return cost_sum / (len(x)*len(t))

def gradient_cost_function(P, x, t, deep_neural_network, activation_function):
    cost_function_grad = grad(cost_function,0)
    return cost_function_grad((P, x, t, deep_neural_network, activation_function))

# Device, this will use GPU if available, else CPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cost_function_pyTorch(model, x, t):
    # make sure x,t are tensors on device
    x = x.to(device)
    t = t.to(device)
    # Create meshgrid then flatten
    T, X = torch.meshgrid(t, x, indexing='ij')  # shape (len(t), len(x))
    T_flat = T.reshape(-1)
    X_flat = X.reshape(-1)
    # IMPORTANT: these must require grad and be the tensors used to compute g_trial
    T_flat = T_flat.reshape(-1).clone().detach().to(device).requires_grad_(True)
    X_flat = X_flat.reshape(-1).clone().detach().to(device).requires_grad_(True)
    g_trial = g_trial_pyTorch(T_flat, X_flat, model)
    # first derivative w.r.t. time for each point (batch)
    ones = torch.ones_like(g_trial, device=device)
    g_t = torch.autograd.grad(
        outputs=g_trial,
        inputs=T_flat,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # first derivative w.r.t. space
    g_x = torch.autograd.grad(
        outputs=g_trial,
        inputs=X_flat,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # second derivative w.r.t. space
    g_xx = torch.autograd.grad(
        outputs=g_x,
        inputs=X_flat,
        grad_outputs=torch.ones_like(g_x, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # PDE residual for e.g. g_t - g_xx (modify if your PDE is different)
    R = g_t - g_xx
    return (R**2).mean()

def cost_function_pyTorch2(model, x, t):
    '''  Cost function for the PINN using PyTorch.
    Parameters:
    -----------
    x : torch.tensor
        Input data of shape (num_coordinates, ) for a single point or (num_coordinates, num_points) for multiple points.
    t : torch.tensor
        Input data of shape (num_coordinates, ) for a single point or (num_coordinates, num_points) for multiple points.
    model : nn.Module
        The neural network model defined using PyTorch.
    Returns:    
        -----------
    cost : torch.tensor'''
    # make sure x,t are tensors on device
    x = x.to(device)
    t = t.to(device)
    # Create meshgrid then flatten
    T, X = torch.meshgrid(t, x, indexing='ij')  # shape (len(t), len(x))
    T_flat = T.reshape(-1)
    X_flat = X.reshape(-1)
    R = g_trial_pyTorch(T_flat, X_flat, model)     # shape (len(t)*len(x),)
    return (R**2).mean()

def g_trial_batch_cost(model, x_batch, t_batch):
    # x_batch and t_batch are 1D tensors of same length (batch_size)
    R = g_trial_pyTorch(t_batch, x_batch, model)   # note order (t,x,model)
    return (R**2).mean()