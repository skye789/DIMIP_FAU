import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size, batch_size, delta):

        # initialize the weights randomly
        self.weights = np.random.rand(output_size, input_size + 1)

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((input_size + 1, batch_size))  # "pre-allocation"

        # allow individual learning rates per hidden layer
        self.delta = delta

    def forward(self, input_tensor):

        # put together the activations from the input_tensor
        # add an additional row of ones to include the bias (such that w^T * x + b becomes w^T * x equivalently)
        # TODO
        

        batch_size = np.shape(input_tensor)[1]
        biases     = np.ones((batch_size,1))
        self.input = np.concatenate((input_tensor.T,biases),axis=1).T 

        output = np.dot(self.weights, self.input)
        # perform the forward pass just by matrix multiplication
        layer_output = output   # TODO
        return layer_output

    def backward(self, error_tensor):

        # update the layer using the learning rate and E * X^T,
        # where E is the error from higher layers and X are the activations stored from the forward pass
        #
        # 1. calculate the error for the next layers using the transposed weights and the error
        # TODO
        gradient_weights = np.dot(error_tensor,self.input.T)
        gradient_tensors = np.dot(self.weights.T,error_tensor)
        # 2. update this layer's weights
        # TODO
        self.weights -= self.delta * gradient_weights 

        # the bias of this layer does not affect the layers before, so delete it from the return value
        error_tensor_new =  gradient_tensors[:np.shape(gradient_tensors)[0]-1,:] # TODO
        return error_tensor_new
