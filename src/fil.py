
##################################################################################
# Import necessary libraries :
##################################################################################
import autograd.numpy as np
from matplotlib import pyplot as plt
import sys, os
import torch
import time
import matplotlib as mpl
from .fig_saver import save_fig
##################################################################################


##################################################################################
# Latex-style plotting settings :
##################################################################################

# --------- Matplotlib style: LaTeX + RevTeX-ish sizes ---------

mpl.rcParams.update({
    # Use LaTeX for all text
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # same as default in LaTeX

    # Base font size ~ RevTeX reprint (10pt)
    "font.size": 20,
    "axes.labelsize": 23,
    "axes.titlesize": 23,
    "xtick.labelsize": 15,
    "ytick.labelsize": 19,
    "legend.fontsize": 12,

    # Slightly nicer layout
    "figure.dpi": 150,
})

# Optional: make sure some LaTeX packages are available for math/units etc.
# (These must also be installed in your local LaTeX distribution.)
mpl.rcParams["text.latex.preamble"] = r"""
\usepackage{amsmath,amssymb,siunitx}
"""


"""
from src.make_config import make_config, create_t_and_x_batch_size
from src.training_PINN_pyTorch import train_PINN_pyTorch
from src.training_PINN import train_PINN_autograd



##########################################################################################
# Import functions :
##########################################################################################
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)
from src.make_config import make_config, create_t_and_x_batch_size
from src.training_PINN_pyTorch import train_PINN_pyTorch
from src.training_PINN import train_PINN_autograd
from src.activation_funcs import tanh
from src.analytical_func import g_analytic
from src.trial_func import trial_function_autograd, trial_function_pyTorch
from src.neural_network import deep_neural_network
from src.neural_network_pyTorch import Net, device
from src.FTCS import ftcs_solution
from src.PINN_solution import PINN_solution, analytical_solution
"""
