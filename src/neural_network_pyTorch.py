##########################################################################################################
# NEURAL NETWORK IMPLEMENTATION :
##########################################################################################################
# Project 3 - FYS-STK4155 :
# Authors : Ingvild Olden and Jenny Guldvog 

##########################################################################################################
# IMPORT NECESSARY PACKAGES :
##########################################################################################################
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Union
#torch.manual_seed(42)
# Device, this will use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################################################
# FEED FORWARD NEURAL NETWORK USING PYTORCH :
##########################################################################################################
class Net(nn.Module):
    def __init__(
        self,
        network: Optional[Union[int, Iterable[int]]] = None,
        activation: Union[str, Iterable[str]] = "tanh",
    ):
        """
        A feedforward deep neural network with multiple hidden layers using PyTorch.

        Parameters
        ----------
        network : int or iterable of int, optional
            If an int is provided, legacy behavior is used: `layers` hidden layers each
            with `hidden` neurons. If an iterable (e.g. [3,5,10]) is provided, it defines
            the number of neurons in each hidden layer explicitly.
            If None (default), legacy behavior: `hidden=64` and `layers=3`.
        activation : str or iterable of str, optional
            Activation function(s) to be used in the hidden layers. If a single string
            is provided, it is applied to all hidden layers. If an iterable is provided,
            it must have the same length as the number of hidden layers, and each entry
            specifies the activation for that layer. Options are "tanh", "relu",
            and "sigmoid". Default is "tanh".
        Returns
        -------
        None
        """
        # This part handles the legacy behavior and input validation, then builds the network
        super().__init__()
        if network is None:
            hidden_sizes = [64] * 3
        elif isinstance(network, int):
            hidden_sizes = [network] * 3
        else:
            hidden_sizes = list(network)
        if isinstance(activation, str):
            activations = [activation] * len(hidden_sizes)
        else:
            activations = list(activation)
            if len(activations) != len(hidden_sizes):
                raise ValueError(
                    f"activation length ({len(activations)}) must match number of hidden layers ({len(hidden_sizes)})"
                )

        def activation_layer(name: str):
            if name == "tanh":
                return nn.Tanh()
            if name == "relu":
                return nn.ReLU()
            if name == "sigmoid":
                return nn.Sigmoid()
            raise ValueError(f"Unknown activation '{name}'")

        dims = [2] + hidden_sizes + [1]
        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(activation_layer(activations[i]))
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)
    def forward(self, t, x, debug=False):
        '''  
        Forward pass through the network.

        Parameters:
        -----------
        t : torch.Tensor
            Time input tensor.
        x : torch.Tensor
            Spatial input tensor.
        debug : bool
            If True, print debug information.

        Returns:
        --------
        torch.Tensor
            Output of the network.

        Example usage:

        model = Net(network=[10, 20, 10], activation="relu")
        t = torch.tensor([0.0, 0.5, 1.0])
        x = torch.tensor([0.0, 0.5, 1.0])
        output = model(t, x)    
        '''
        inp = torch.stack([t, x], dim=-1)
        out = self.net(inp)
        if debug:
            print("Input to neural network (pyTorch): ")
            print("inp.shape =",inp.shape)
            print("num_points:", inp.shape[0])
            print("num_coordinates:", inp.shape[1])
            print("Number of hidden layers:", len(self.net)//2 - 1)
            print("Neural network output (pyTorch): ")
            print("out.shape =", out.shape)
        return out.squeeze(-1)
