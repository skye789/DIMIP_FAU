import numpy as np


class SoftMax:

    def __init__(self, categories, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((categories, batch_size))  # "pre-allocation"

    def forward(self, input_tensor):

        # store the activations from the input_tensor
        # self.activations = np.copy(input_tensor)

        # apply SoftMax to the scores: e(x_i) / sum(e(x))
        
        max_x = np.max(input_tensor,axis=0)        
        xk_mat = np.zeros(np.shape(input_tensor))
        for i in range(len(max_x)):
            xk_mat[:,i] = max_x[i]

        sum_xj = np.sum(np.exp(input_tensor-xk_mat),axis=0)        
        sum_mat = np.zeros(np.shape(input_tensor))
        for i in range(len(sum_xj)):
            sum_mat[:,i] = sum_xj[i]
        
        self.y_hat = np.exp(input_tensor-xk_mat) / sum_mat
        print(input_tensor.shape)
        print(np.sum(self.y_hat))
        return self.y_hat


    def backward(self, label_tensor):

        # error_tensor = np.copy(self.activations)
        #  Given:
        #  - the labels are one-hot vectors
        #  - the loss is cross-entropy (as implemented below)
        # Idea:
        # - decrease the output everywhere except at the position where the label is correct
        # - implemented by increasing the output at the position of the correct label
        # Hint:
        # - do not let yourself get confused by the terms 'increase/decrease'
        # - instead consider the effect of the loss and the signs used for the backward pass

        # TODO
        # ...
        return self.y_hat-label_tensor

    def loss(self, label_tensor):

        # iterate over all elements of the batch and sum the loss
        # TODO
        # ... # loss is the negative log of the activation of the correct position
        y_hat = self.y_hat * label_tensor
        loss = np.sum(-1 * np.log(y_hat + np.finfo(float).eps))
        print(loss)

        return loss
