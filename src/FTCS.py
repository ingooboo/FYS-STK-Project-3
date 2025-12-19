import autograd.numpy as np

def ftcs_solution(x, t):
    # dx is created from x-array, and is uniform
    dx = float(x[1] - x[0])
    # number of spatial points
    nx = len(x)
    # final time
    t_end = float(t[-1])
    # stable time step; adjust to hit t_end exactly
    dt_stable = 0.5 * dx**2 # stability condition for FTCS scheme, alpha <= 0.5
    # number of time steps
    nt = int(np.ceil(t_end / dt_stable)) + 1
    # actual time step
    dt = t_end / (nt - 1)
    # diffusion coefficient, used in scheme, for clarity, 
    alpha = dt / dx**2
    # initial + boundaries
    u = np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0
    sol = np.zeros((nt, nx))
    sol[0] = u
    for n in range(1, nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + alpha * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new
        sol[n] = u
    t_grid = np.linspace(0.0, t_end, nt)
    # interpolate in time so we align with the chosen t-array
    sol_interp = np.vstack([np.interp(t, t_grid, sol[:, i]) for i in range(nx)])  # (nx, len(t))
    return sol_interp  # shape (nx, len(t)) to match g_dnn_ag
