from __future__ import annotations

import autograd.numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from .fig_saver import save_fig
from .neural_network_pyTorch import Net, device
from .trial_func import trial_function_pyTorch
from .analytical_func import g_analytic
from .cost_func import cost_function_pyTorch
from .training_PINN_pyTorch import train_PINN_pyTorch


# Global-ish context so notebook code can set shared variables once instead of redefining
_ctx = {}
_saved_analytical_once = False


def set_context(*, x, t, x_torch, t_torch, Nx, Nt, epochs, learning_rate, seed, batch_size, stop_delta=0.0, stop_patience=None):
    """
    Store common tensors/params so helpers can be used without notebook globals.

    Call this once in the notebook after you have created x, t, x_torch, t_torch, etc.
    """
    _ctx.update(
        dict(
            x=x,
            t=t,
            x_torch=x_torch,
            t_torch=t_torch,
            Nx=Nx,
            Nt=Nt,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            batch_size=batch_size,
            stop_delta=stop_delta,
            stop_patience=stop_patience,
        )
    )


def _require_ctx(keys):
    missing = [k for k in keys if k not in _ctx]
    if missing:
        raise NameError(
            f"Missing context {missing}. Call set_context(...) first or pass the needed arguments explicitly."
        )

def _format_label(item):
    return '-'.join(map(str, item)) if isinstance(item, (list, tuple)) else str(item)

def _architecture_title_label(num_hidden_neurons):
    if isinstance(num_hidden_neurons, (list, tuple)) and len(num_hidden_neurons) > 0:
        unique_widths = set(num_hidden_neurons)
        if len(unique_widths) == 1:
            return f"({len(num_hidden_neurons)},{num_hidden_neurons[0]})"
    return _format_label(num_hidden_neurons)

def plot_an_nn_diff(
    g_dnn_ag,
    G_analytical,
    diff_ag,
    activation_func,
    num_hidden_neurons,
    x=None,
    t=None,
    save_prefix=None,
    save_analytical_once=True,
):
    activation_label = _format_label(activation_func)
    architecture_label = _format_label(num_hidden_neurons)
    title_architecture_label = _architecture_title_label(num_hidden_neurons)
    prefix = save_prefix or f"{activation_label}_{architecture_label}"
    if x is None or t is None:
        _require_ctx(["x", "t"])
        x = _ctx["x"]
        t = _ctx["t"]

    plt.figure(figsize=(6,5))
    plt.imshow(g_dnn_ag,
           origin='lower',
           extent=[x.min(), x.max(), t.min(), t.max()],
           aspect='auto',
           cmap='plasma')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'PINN solution: {activation_label}, {title_architecture_label}')
    plt.colorbar()
    save_fig(f'pinn_solution_{prefix}')
    plt.show()

    plt.figure(figsize=(5,5))
    plt.imshow(G_analytical,
           origin='lower',
           extent=[x.min(), x.max(), t.min(), t.max()],
           aspect='auto',
           cmap='plasma')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Analytical')
    plt.colorbar()
    global _saved_analytical_once
    if save_analytical_once:
        if not _saved_analytical_once:
            save_fig('analytical_solution')
            _saved_analytical_once = True
    else:
        save_fig(f'analytical_solution_{prefix}')
    plt.show()


    plt.figure(figsize=(5,5))
    v = float(np.max(np.abs(diff_ag)))
    v = v if v > 0 else 1e-12
    linthresh = max(1e-10, v * 1e-3)
    norm = SymLogNorm(linthresh=linthresh, vmin=-v, vmax=v)
    plt.imshow(diff_ag,
           origin='lower',
           extent=[x.min(), x.max(), t.min(), t.max()],
           aspect='auto',
           cmap='RdBu_r',
           norm=norm)
    plt.xlabel('t')
    plt.ylabel('x')
    title_label = f"{activation_label}, {title_architecture_label}"
    plt.title(f'Difference: best PINN - Analytical')
    cbar = plt.colorbar()
    save_fig(f'pinn_minus_analytical_{prefix}')
    plt.show()


def comp_max_abs_diff(
    Nx=None,
    Nt=None,
    x_torch=None,
    t_torch=None,
    P=None,
    num_hidden_neurons=None,
    activation_func=None,
):
    if P is None or num_hidden_neurons is None or activation_func is None:
        raise ValueError("Provide P, num_hidden_neurons and activation_func.")
    # Use context defaults if not provided
    if Nx is None or Nt is None or x_torch is None or t_torch is None:
        _require_ctx(["Nx", "Nt", "x_torch", "t_torch"])
        Nx = Nx if Nx is not None else _ctx["Nx"]
        Nt = Nt if Nt is not None else _ctx["Nt"]
        x_torch = x_torch if x_torch is not None else _ctx["x_torch"]
        t_torch = t_torch if t_torch is not None else _ctx["t_torch"]
    # Store the results
    g_dnn_ag = np.zeros((Nx, Nt))
    G_analytical = np.zeros((Nx, Nt))
    # Build and load the model once instead of inside the grid loops
    model = Net(network=num_hidden_neurons, activation=activation_func).to(device)
    model.load_state_dict(P)
    model.eval()
    with torch.no_grad():
        for i, x_ in enumerate(x_torch):
            for j, t_ in enumerate(t_torch):
                point = np.array([x_, t_])
                g_dnn_ag[i,j] = trial_function_pyTorch(x_, t_, model).detach().cpu().numpy()
                G_analytical[i,j] = g_analytic(point)
        # Find the map difference between the analytical and the computed solution
        diff_ag = g_dnn_ag - G_analytical
        max_abs_diff = np.max(np.abs(diff_ag))
        print('Max absolute difference between the analytical solution and the network: %g'%max_abs_diff)
    # Find the final cost value (requires grad tracking)
    final_cost = cost_function_pyTorch(x_torch, t_torch, model).item()
    print('Final cost: {:.6e}'.format(final_cost))
    return max_abs_diff, final_cost, g_dnn_ag, G_analytical, diff_ag


def plot_cost_history(history, activation_function, num_hidden_neurons, save_name=None):
    # Plot cost history
    iters = history['epoch']
    costs = history['cost']
    plt.figure(figsize=(5,5))
    plt.plot(iters, costs, '-o', markersize=4)
    plt.xlabel('epoch')
    plt.ylabel('Cost')
    plt.title(f'Training cost - ADAM SGD {activation_function} - {num_hidden_neurons}')
    plt.grid(True)
    plt.tight_layout()
    activation_label = _format_label(activation_function)
    architecture_label = _format_label(num_hidden_neurons)
    save_fig(save_name or f'training_cost_{activation_label}_{architecture_label}')
    plt.show()


# Heatmap helpers for architecture sweeps
def build_architecture(width, depth):
    return [width] * depth


def evaluate_architecture(
    width,
    depth,
    activation_function,
    *,
    x_torch=None,
    t_torch=None,
    Nx=None,
    Nt=None,
    epochs=None,
    learning_rate=None,
    seed=None,
    batch_size=None,
    stop_delta=None,
    stop_patience=None,
):
    # Pull from context if not provided explicitly
    if any(v is None for v in (x_torch, t_torch, Nx, Nt, epochs, learning_rate, seed, batch_size, stop_delta, stop_patience)):
        _require_ctx(["x_torch", "t_torch", "Nx", "Nt", "epochs", "learning_rate", "seed", "batch_size", "stop_delta", "stop_patience"])
        x_torch = x_torch or _ctx["x_torch"]
        t_torch = t_torch or _ctx["t_torch"]
        Nx = Nx or _ctx["Nx"]
        Nt = Nt or _ctx["Nt"]
        epochs = epochs or _ctx["epochs"]
        learning_rate = learning_rate or _ctx["learning_rate"]
        seed = seed or _ctx["seed"]
        batch_size = batch_size or _ctx["batch_size"]
        stop_delta = stop_delta if stop_delta is not None else _ctx["stop_delta"]
        stop_patience = stop_patience if stop_patience is not None else _ctx["stop_patience"]

    num_hidden_neurons = build_architecture(width, depth)
    P, history = train_PINN_pyTorch(
        x_torch,
        t_torch,
        num_hidden_neurons,
        epochs,
        learning_rate,
        activation_function,
        seed,
        optimization_method="SGD-adam",
        batch_size=batch_size,
        shuffle=True,
        replacement=False,
        verbose=False,
        debug=False,
        stop_patience=stop_patience,
        stop_delta=stop_delta,
    )
    max_abs_diff, final_cost, g_dnn_ag, G_analytical, diff_ag = comp_max_abs_diff(
        Nx, Nt, x_torch, t_torch, P, num_hidden_neurons, activation_function
    )
    return final_cost, max_abs_diff

def plot_metric_heatmap(metric_grid, widths, depths, title, cbar_label, cmap='YlOrRd', log_scale=False, save_name=None):
    data = np.array(metric_grid, dtype=float)
    eps = 1e-12
    data_plot = np.log10(np.clip(data, eps, None)) if log_scale else data
    cbar_label_use = f'log10({cbar_label})' if log_scale else cbar_label

    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(data_plot, origin='lower', cmap=cmap)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Number of nodes per layer')
    ax.set_ylabel('Number of hidden layers')
    ax.set_title(title.replace('|error|', r'$|error|$'))
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label_use.replace('|error|', r'$|error|$'))
    # Annotate with original values
    mean_val = data_plot.mean() if data_plot.size > 0 else 0.0
    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            val = data[i, j]
            ax.text(j, i, f'{val:.2e}', ha='center', va='center',
                    color='black' if data_plot[i, j] < mean_val else 'white', fontsize=12)
    plt.tight_layout()
    # Avoid pipes in filenames (e.g., "Max |error|") to keep paths shell-friendly
    slug = title.lower().replace('|', '').replace(' ', '_').replace('-', '_')
    save_fig(save_name or f'heatmap_{slug}', fig=fig)
    plt.show()
