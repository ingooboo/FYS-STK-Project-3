"""Autograd-based feedforward network utilities."""

##########################################################################################################
# IMPORT NECESSARY PACKAGES :
##########################################################################################################
import autograd.numpy as np

##########################################################################################################
# FEED FORWARD NEURAL NETWORK USING AUTOGRAD :
##########################################################################################################
def deep_neural_network(deep_params, 
                        input_data, 
                        activation_func,
                        debug=False):
    """Run a feedforward pass through the network.

    Args:
        deep_params: List of weight matrices including bias rows.
        input_data: Array of shape (num_coordinates,) or (num_coordinates, num_points).
        activation_func: Activation function applied to hidden layers.
        debug: If True, print debug information.

    Returns:
        Network output array.
    """
    # Reshape input x to be of shape (num_coordinates, num_points)
    # Because x can be given as a single point (num_coordinates, ) or multiple points (num_coordinates, num_points)
    num_coordinates = np.size(input_data,0)
    # Reshape x accordingly, such that it has shape (num_coordinates, num_points)
    x = input_data.reshape(num_coordinates,-1)
    # num_points is the number of input points
    num_points = np.size(x,1)
    # Feedforward through the hidden layers, where N_hidden is the number of hidden layers
    N_hidden = len(deep_params) - 1 
    # Input layer:
    x_input = x
    # Initialize x_prev to be the input layer
    x_prev = x_input
    if debug:
        print("Input to neural network:")
        print("x_prev.shape =", x_prev.shape)
        print("num_points:", num_points)
        print("num_coordinates:", num_coordinates)
        print("Number of hidden layers:", N_hidden)
    # Loop over the hidden layers
    for l in range(N_hidden):
        # From the list of parameters deep_params; find the correct weigths and bias for this layer
        w_hidden = deep_params[l]
        # Add a row of ones to include bias, shape becomes (num_coordinates + 1, num_points)
        x_prev = np.concatenate((np.ones((1,num_points)), x_prev ), axis = 0)
        # Compute the output from this layer
        z_hidden = np.matmul(w_hidden, x_prev)
        # Apply activation function
        x_hidden = activation_func(z_hidden)
        # alternatively:
        #z_hidden = x_hidden @ w_hidden.T + np.ones((num_points,1)) @ w_hidden[0:1,:].T
        # Update x_prev such that next layer can use the output from this layer
        x_prev = x_hidden
    # Output layer:
    # Get the weights and bias for this layer
    w_output = deep_params[-1]
    # Include bias:
    x_prev = np.concatenate((np.ones((1,num_points)), x_prev), axis = 0)
    # Compute the output from the output layer, no activation function applied here
    z_output = np.matmul(w_output, x_prev)
    #return z_output[0][0]
    if debug:
        print("Output from neural network:")
        print(z_output.shape)
    return z_output.squeeze()

class DeepNeuralNetwork:
    """Feedforward network with multiple hidden layers.

    Args:
        deep_params: List of weight matrices including bias rows.
        activation_func: Activation function for hidden layers.
    """

    def __init__(self, deep_params, activation_func):
        # Expect deep_params to be a list of 2D arrays
        self.deep_params = deep_params
        self.activation  = activation_func
        # Map string activations to concrete callables.
        if self.activation == 'tanh':
            self.activation = np.tanh
        elif self.activation == 'sigmoid':
            self.activation = lambda z: 1/(1 + np.exp(-z))
        elif self.activation == 'relu':
            self.activation = lambda z: np.maximum(0, z)
 
    def set_params(self, deep_params):
        """Replace the network parameters.

        Args:
            deep_params: New list of weight matrices.
        """
        self.deep_params = deep_params

    def get_params(self):
        """Return the network parameters.

        Returns:
            List of weight matrices.
        """
        return self.deep_params

    def forward(self, input_data, debug=False):
        """Perform a feedforward pass.

        Args:
            input_data: Array of shape (num_coordinates,) or (num_coordinates, num_points).
            debug: If True, print intermediate shapes and info.

        Returns:
            Output array with singleton dimensions removed.
        """
        # Ensure shape (num_coordinates, num_points)
        num_coordinates = np.size(input_data, 0)
        x = input_data.reshape(num_coordinates, -1)   # shape (num_coordinates, num_points)
        num_points = np.size(x, 1)
        # Number of hidden layers = total layers - 1 (output)
        N_hidden = len(self.deep_params) - 1
        # Input layer
        x_prev = x
        if debug:
            print("Input to neural network (autograd):")
            print("x_prev.shape =", x_prev.shape)
            print("num_points:", num_points)
            print("num_coordinates:", num_coordinates)
            print("Number of hidden layers:", N_hidden)
        # Loop through hidden layers
        for l in range(N_hidden):
            w_hidden = self.deep_params[l]  # expected shape: (n_neurons_l, num_coordinates + 1)
            # Add bias row of ones: shape becomes (num_coordinates + 1, num_points)
            x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)
            # Linear transform
            z_hidden = np.matmul(w_hidden, x_prev)
            # Activation
            x_hidden = self.activation(z_hidden)
            # Update for next layer
            x_prev = x_hidden
        # Output layer (no activation)
        w_output = self.deep_params[-1]
        x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)
        z_output = np.matmul(w_output, x_prev)
        if debug:
            print("Output from neural network (autograd):")
            print("z_output.shape =", z_output.shape)
        return z_output.squeeze()
    
    # Optional convenience alias
    def __call__(self, input_data, debug=False):
        """Alias for forward()."""
        return self.forward(input_data, debug=debug)
