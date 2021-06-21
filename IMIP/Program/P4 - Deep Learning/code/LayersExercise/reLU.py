import numpy as np


class ReLU:

    def __init__(self, input_size, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((input_size, batch_size))  # "pre-allocation"

    def forward(self, input_tensor):

        # store the activations from the input_tensor
        # TODO

        input_tensor[input_tensor<=0] = 0

        ReLU_respect_X = input_tensor.copy()
        ReLU_respect_X[ReLU_respect_X>0] = 1
        self.ReLU_respect_X = ReLU_respect_X

        # # the output is max(0, activation)
        # layer_output = np.max(0,input_tensor)  # TODO
        return input_tensor

    def backward(self, error_tensor):

        # the gradient is zero whenever the activation is negative

        return error_tensor * self.ReLU_respect_X
