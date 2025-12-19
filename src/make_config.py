import autograd.numpy as np
import torch

def make_config(Nx,Nt, num_hidden_neurons, opt_met, epochs, learning_rate, batch_fraction, batch_size, seed, verbose):
    ##################################################################################
    # CONFIGURATION INFORMATION : 
    ##################################################################################
    configuration_information = {
        'Grid resolution         ': f'Nx={Nx}, Nt={Nt}',
        'Network architecture    ': f'{num_hidden_neurons}',
        'Gradient decent method  ': opt_met,
        'Learning rate           ': f'{learning_rate}',
        'Epochs                  ': f'{epochs}',
        'Batch fraction          ': f'{batch_fraction}',
        'Batch size              ': f'{batch_size}',
        'Seed                    ': f'{seed}'
    }
    config_info_formatted = '\n'.join([f'{key}: {value}' for key, value in configuration_information.items()])
    config_info = f'--------------------------------------\nConfiguration information :\n\n{config_info_formatted}\n--------------------------------------'
    if verbose:
        print(config_info)
    return configuration_information

def create_t_and_x_batch_size(Nx,Nt,batch_fraction=0.1):
    ##################################################################################
    # RANGE FOR x (lenght of rod, always 0-1) AND t (time, def 0-1) : 
    ##################################################################################
    t = np.linspace(0.0, 1.0, Nt)
    x = np.linspace(0.0, 1.0, Nx)   
    t_torch = torch.linspace(0.0, 1.0, Nt)
    x_torch = torch.linspace(0.0, 1.0, Nx)
    # ONLY FOR STOCHASTIC GRADIENT DECENT : 
    batch_size     = int(Nx*Nt*batch_fraction)
    return t, x, t_torch, x_torch, batch_size

##########################################################################################
# FUNCTION TO SET UP PARAMETERS :
##########################################################################################
def set_up_test_grid_res(Nx,
                         Nt,
                         opt_met,
                         num_hidden_neurons,
                         epochs,
                         learning_rate,
                         batch_fraction=0.1,
                         seed=None,
                         verbose=False):
    ##################################################################################
    # MAKE X AND T : 
    ##################################################################################
    t, x, t_torch, x_torch, batch_size = create_t_and_x_batch_size(Nx,
                                                                   Nt,
                                                                   batch_fraction)
    ##################################################################################
    # CONFIGURATION INFORMATION : 
    ##################################################################################
    config = make_config(Nx,
                         Nt, 
                         num_hidden_neurons,
                         opt_met, 
                         epochs, 
                         learning_rate, 
                         batch_fraction,
                         batch_size,
                         seed,
                         verbose=verbose)
    return t, x, t_torch, x_torch, batch_size, config
