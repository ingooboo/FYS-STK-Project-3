##################################################################################
# Import necessary libraries :
##################################################################################
import autograd.numpy as np
from .analytical_func import g_analytic
from .trial_func import trial_function_autograd, trial_function_pyTorch
from .neural_network_pyTorch import Net, device

def PINN_solution(Nx,Nt,x,t,model_imp,P,num_hidden_neurons,activation_func,autograd_or_pytorch):
    temp_u = np.zeros((Nx, Nt))
    for i, x_ in enumerate(x):
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            if autograd_or_pytorch == 'pytorch':
                model = model_imp(network=num_hidden_neurons, activation=activation_func).to(device)
                model.load_state_dict(P)
                temp_u[i,j] = trial_function_pyTorch(x_, t_, model).detach().cpu().numpy()
            if autograd_or_pytorch == 'autograd':
                temp_u[i,j] = trial_function_autograd(x_, t_, model_imp, P, activation_function=activation_func)
    return temp_u

def analytical_solution(Nx,Nt,x,t):
    analytical_temp = np.zeros((Nx, Nt))
    for i, x_ in enumerate(x):
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            analytical_temp[i,j] = g_analytic(point)
    return analytical_temp