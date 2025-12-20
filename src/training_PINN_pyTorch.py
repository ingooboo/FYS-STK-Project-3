"""Training utilities for the PyTorch-based PINN."""

##########################################################################################################
# IMPORT NECESSARY PACKAGES :
##########################################################################################################
import torch
import math
import torch.nn as nn
import copy
from .neural_network_pyTorch import Net
from .cost_func import cost_function_pyTorch as cost_function
# Device, this will use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################################
# TRAIN THE DEEP NEURAL NETWORK TO SOLVE THE PDE :
##########################################################################################################
def train_PINN_pyTorch(x,
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
                       debug=False,
                       stop_patience=None,
                       stop_delta=0.0):
    """Train the PyTorch PINN with GD/Adam/SGD-Adam.

    Args:
        x: Spatial grid tensor.
        t: Time grid tensor.
        num_neurons: Hidden layer widths.
        epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        activation_function: Activation function name.
        seed: RNG seed.
        optimization_method: 'GD', 'GD-adam', or 'SGD-adam'.
        batch_size: Mini-batch size for SGD-Adam.
        shuffle: If True, shuffle batches each epoch.
        replacement: If True, sample batches with replacement.
        verbose: If True, print training progress.
        debug: If True, print debug information from model/cost.
        stop_patience: Early-stopping patience in epochs.
        stop_delta: Early-stopping improvement threshold.

    Returns:
        Tuple of (state_dict, history).
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    # Make the model
    model = Net(network=num_neurons, activation=activation_function).to(device)
    if optimization_method == 'GD':
        # Plain gradient descent optimizer (not stochastic or mini-batch)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimization_method == 'GD-adam':
        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimization_method == 'SGD-adam':
        # Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # prepare x and t as 1D tensors
    x_t = x.clone().detach().to(device).view(-1)
    t_t = t.clone().detach().to(device).view(-1)
    if optimization_method == 'SGD-adam':
        if batch_size is None:
            raise ValueError("batch_size must be provided when using 'SGD-adam' optimizer")
        # build full meshgrid coordinates (flattened) once
        T_full, X_full = torch.meshgrid(t_t, x_t, indexing='ij')
        T_flat = T_full.reshape(-1)
        X_flat = X_full.reshape(-1)
        num_points = T_flat.shape[0]
    # --- begin: added history collection ---
    history = {'epoch': [], 'cost': []}
    init_cost = cost_function(x_t, t_t, model, debug).item()
    history['epoch'].append(0)
    history['cost'].append(init_cost)
    # Track best model if early stopping is requested
    best_state = None
    best_cost = init_cost
    if stop_patience is not None:
        best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    # --- end: added history collection ---
    print('Initial cost: ',cost_function(x_t, t_t, model, debug).item())
    if optimization_method == 'GD' or optimization_method == 'GD-adam':
        for it in range(1, epochs + 1):
            optimizer.zero_grad()
            cost = cost_function(x_t, t_t, model, debug)
            cost.backward()
            optimizer.step()
            # --- begin: added history collection ---
            history['epoch'].append(it)
            history['cost'].append(cost.item())
            # --- end: added history collection ---
            if verbose:
                print('Iteration: {}, Cost: {:.6e}'.format(it, cost.item()))
            if stop_patience is not None:
                cost_val = cost.item()
                if best_cost - cost_val > stop_delta:
                    best_cost = cost_val
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= stop_patience:
                        if verbose:
                            print(f'Early stopping at epoch {it} (no improvement for {stop_patience} steps).')
                        break
    iter_count = 0
    # Helper: yield minibatch index ranges (no replacement) for one epoch
    def epoch_minibatch_indices_no_replacement(order, batch_size):
        n_batches = int(math.ceil(num_points / float(batch_size)))
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, num_points)
            yield order[start:end]
    if optimization_method == 'SGD-adam':
        # For replacement mode choose similar number of batches per epoch
        batches_per_epoch_replacement = int(math.ceil(num_points / float(batch_size)))
        # Main epoch loop
        for epoch in range(1, epochs + 1):
            # Sample indices for collocation points in the flattened meshgrid
            if replacement:
                # torch.randint with three args is safe and supported:
                # low (inclusive), high (exclusive), size tuple
                indices = torch.randint(0, num_points, (batch_size,), device=device)
                t_batch = T_flat[indices]
                x_batch = X_flat[indices]
                optimizer.zero_grad()
                cost = cost_function(x_batch, t_batch, model, debug)
                cost.backward()
                optimizer.step()
                iter_count += 1
            else:
                # no replacement: shuffle or fixed order
                if shuffle:
                    order = torch.randperm(num_points, device=device)
                else:
                    order = torch.arange(num_points, device=device)
                # iterate contiguous slices of the permuted/ordered flattened points
                for idx_range in epoch_minibatch_indices_no_replacement(order, batch_size):
                    # idx_range is a torch tensor containing indices for this batch
                    # ensure it's a 1D tensor of dtype long (it already will be)
                    if idx_range.numel() == 0:
                        continue
                    # If batch_size > num_points we fall back to with-replacement sampling for that batch
                    if idx_range.numel() < batch_size and batch_size > num_points:
                        indices = torch.randint(0, num_points, (batch_size,), device=device)
                    else:
                        indices = idx_range
                    t_batch = T_flat[indices]
                    x_batch = X_flat[indices]
                    optimizer.zero_grad()
                    cost = cost_function(x_batch, t_batch, model, debug)
                    cost.backward()
                    optimizer.step()
                    iter_count += 1
            # End of epoch logging (compute full-dataset cost using cost_function_pyTorch)
            full_cost = cost_function(x_t, t_t, model, debug)
            if verbose:
                print(f'Epoch {epoch:4d} Iter {iter_count:6d} Cost {full_cost:.6e}')
            history['epoch'].append(epoch)
            history['cost'].append(full_cost.item())
            if stop_patience is not None:
                full_cost_val = full_cost.item()
                if best_cost - full_cost_val > stop_delta:
                    best_cost = full_cost_val
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= stop_patience:
                        if verbose:
                            print(f'Early stopping at epoch {epoch} (no improvement for {stop_patience} epochs).')
                        break
    if stop_patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    print('Final cost: ',cost_function(x_t, t_t, model, debug).item())
    return model.state_dict(), history
