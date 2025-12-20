"""PyTorch feedforward network used by the PINN."""

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
        """Initialize a feedforward network with configurable widths and activations.

        Args:
            network: Hidden-layer widths or single width for legacy behavior.
            activation: Activation name(s) for hidden layers.

        Raises:
            ValueError: If activation list length does not match hidden layers.
        """
        # This part handles the legacy behavior and input validation, then builds the network
        super().__init__()
        # Normalize network width spec to a list.
        if network is None:
            hidden_sizes = [64] * 3
        elif isinstance(network, int):
            hidden_sizes = [network] * 3
        else:
            hidden_sizes = list(network)
        # Normalize activations to a list aligned with hidden sizes.
        if isinstance(activation, str):
            activations = [activation] * len(hidden_sizes)
        else:
            activations = list(activation)
            if len(activations) != len(hidden_sizes):
                raise ValueError(
                    f"activation length ({len(activations)}) must match number of hidden layers ({len(hidden_sizes)})"
                )

        def activation_layer(name: str):
            # Map activation names to PyTorch modules.
            if name == "tanh":
                return nn.Tanh()
            if name == "relu":
                return nn.ReLU()
            if name == "sigmoid":
                return nn.Sigmoid()
            raise ValueError(f"Unknown activation '{name}'")

        dims = [2] + hidden_sizes + [1]
        modules = []
        # Build hidden layers with activations.
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(activation_layer(activations[i]))
        # Output layer (linear).
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*modules)
    def forward(self, t, x, debug=False):
        """Run a forward pass.

        Args:
            t: Time input tensor.
            x: Spatial input tensor.
            debug: If True, print debug information.

        Returns:
            Network output as a 1D tensor.
        """
        # Stack inputs into [t, x] pairs for the linear layers.
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
