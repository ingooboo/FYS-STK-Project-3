import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from .fig_saver import save_fig

'''def plot_imshow(x, t, solution, cmap, title, diff):
    if diff:
        v = np.max(np.abs(solution))
        v = v if v > 0 else 1e-12
        plt.imshow(solution,
                origin='lower',
                extent=[x.min(), x.max(), t.min(), t.max()],
                aspect='auto',
                cmap=cmap,
                vmin=-v,
                vmax=v)
    else:
        plt.imshow(solution,
               origin='lower',
               extent=[x.min(), x.max(), t.min(), t.max()],
               aspect='auto',
               cmap=cmap)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title)
    plt.colorbar()
    plt.show()'''

def _sci_fmt(x, pos):
    """Format numbers as compact scientific notation like 1e-3 or 3e-4."""
    if x == 0:
        return "0"
    # Use lower-case 'e' style, and remove leading + in exponent
    s = f"{x:.0e}" if abs(x) < 1e-2 else f"{x:.0e}"
    # Convert "1e-03" -> "1e-3" and "3e-04" -> "3e-4"
    s = s.replace("E", "e").replace("+", "")
    # Remove unnecessary leading zeros in exponent (matplotlib already does this usually)
    return s

def plot_imshow(x, t, solution, cmap, title, diff, save_name=None):
    """
    x, t: 1D arrays used for extent: [x.min(), x.max(), t.min(), t.max()]
    solution: 2D array with shape (len(t), len(x)) or (ny, nx)
    diff: if True, use symmetric vmin/vmax and custom colorbar ticks/formatting
    save_name: optional filename (without path) to save using the shared helper
    """
    # create image
    if diff:
        v = np.max(np.abs(solution))
        v = v if v > 0 else 1e-12
        m = plt.imshow(
            solution,
            origin='lower',
            extent=[x.min(), x.max(), t.min(), t.max()],
            aspect='auto',
            cmap=cmap,
            vmin=-v,
            vmax=v
        )
    else:
        m = plt.imshow(
            solution,
            origin='lower',
            extent=[x.min(), x.max(), t.min(), t.max()],
            aspect='auto',
            cmap=cmap
        )

    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)

    cbar = plt.colorbar(m)

    if diff:
        # Define ticks at symmetric fractions of ±v (customize fractions as you like)
        fracs = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # => v, v/3, v/10
        pos_ticks = v * fracs
        neg_ticks = -pos_ticks[::-1]
        ticks = np.hstack([neg_ticks, [0.0], pos_ticks])
        # Ensure ticks lie within [-v, v]
        ticks = ticks[(ticks >= -v) & (ticks <= v)]
        cbar.set_ticks(ticks)
        cbar.formatter = FuncFormatter(_sci_fmt)
        cbar.update_ticks()

    if save_name:
        save_fig(save_name, fig=plt.gcf())

    plt.show()


def plot_lines(PINN_sol, FTCS_sol, Analytic_sol, save_name=None):
    x_positions = [0.5, 0.625, 0.75, 0.875, 0.1]          # list of x positions for right plot
    x_idx = [np.argmin(np.abs(x - xi)) for xi in x_positions]
    # linestyles for the three solutions
    linestyles = {'PINN': '--', 'FTCS': ':', 'Analytical': '-'}
    # solution linestyle proxies
    sol_proxies = [
        Line2D([0], [0], color='k', linestyle=linestyles['PINN'], lw=2, label='PINN'),
        Line2D([0], [0], color='k', linestyle=linestyles['FTCS'], lw=2, label='FTCS'),
        Line2D([0], [0], color='k', linestyle=linestyles['Analytical'], lw=2, label='Analytical'),
    ]
    cmap_x = plt.get_cmap('cool')
    n_x_colors = len(x_idx)
    x_colors = [cmap_x(i / max(1, n_x_colors - 1)) for i in range(n_x_colors)]
    # -----------------------
    # FIGURE 2: G vs t at selected x positions
    # -----------------------
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for ci, ix in enumerate(x_idx):
        color = x_colors[ci]
        pinn_t = PINN_sol[ix, :]
        ftcs_t = FTCS_sol[ix, :]
        ana_t = Analytic_sol[ix, :]
        ax.plot(t, pinn_t, linestyle=linestyles['PINN'], color=color, linewidth=2.0)
        ax.plot(t, ana_t, linestyle=linestyles['Analytical'], color=color, linewidth=3.5)
        ax.plot(t, ftcs_t, linestyle=linestyles['FTCS'], color='k', linewidth=1.5)
    ax.set_title('u(t) at selected x positions (colors = x, linestyles = solution)')
    ax.set_xlabel('t')
    ax.set_ylabel('temp')
    ax.grid(True)
    # Legend proxies for x-color and solution linestyles
    x_proxies = [Line2D([0], [0], color=x_colors[i], lw=3) for i in range(n_x_colors)]
    x_labels = [f'x={x[ix]:.3g}' for ix in x_idx]
    sol_proxies2 = sol_proxies  # reuse
    # place color legend to the right, linestyles top-left
    leg2 = ax.legend(x_proxies, x_labels, title='x positions', loc='center left',
                    bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
    ax.add_artist(leg2)
    ax.legend(sol_proxies2, [p.get_label() for p in sol_proxies2], loc='upper right', framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_name:
        save_fig(save_name, fig=plt.gcf())

    plt.show()
