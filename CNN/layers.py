import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    :param W: np array - weights
    :param reg_strength: float value
    :return:
        loss: single value - l2 regularization loss
        grad: np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength * np.sum(np.square(W))
    grad = W * 2 * reg_strength
    return loss, grad


def softmax(predictions):
    """
    Computes probabilities from scores
    :param predictions: np array, shape is either (N) or (batch_size, N) - classifier output
    :return:
        probs: np array of the same shape as predictions - probability for every class, 0..1
    """
    if predictions.ndim == 1:
        predictions_exp = np.exp(predictions - np.max(predictions))
        return predictions_exp / np.sum(predictions_exp)
    else:
        predictions_exp = np.exp(predictions - np.max(predictions, axis=1).reshape(-1, 1))
        return predictions_exp / np.sum(predictions_exp, axis=1).reshape(-1, 1)


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss
    :param probs: np array, shape is either (N) or (batch_size, N) - probabilities for every class
    :param target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)
    :return:
        loss: single value
    """
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    else:
        return np.mean(-np.log(probs[np.arange(probs.shape[0]), target_index]))


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions, including the gradient
    :param predictions: np array, shape is either (N) or (batch_size, N) - classifier output
    :param target_index: np array of int, shape is (1) or (batch_size) - index of the true class for given sample(s)
    :return:
        loss: single value - cross-entropy loss
        dprediction: np array same shape as predictions - gradient of predictions by loss value
    """
    soft_max = softmax(predictions)
    loss = cross_entropy_loss(soft_max, target_index)
    if predictions.ndim == 1:
        soft_max[target_index] -= 1
    else:
        soft_max[np.arange(soft_max.shape[0]), target_index] -= 1
        soft_max /= soft_max.shape[0]
    dprediction = soft_max
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.zero = None

    def forward(self, X):
        self.zero = np.nonzero(X < 0)
        res = np.array(X)
        res[self.zero] = 0
        return res

    def backward(self, d_out):
        """
        Backward pass
        :param d_out: np array (batch_size, num_features) - gradient of loss function with respect to output
        :return:
            d_result: np array (batch_size, num_features) - gradient with respect to input
        """
        d_result = np.array(d_out)
        d_result[self.zero] = 0
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = np.array(X)
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        :param d_out: np array (batch_size, n_output) - gradient of loss function with respect to output
        :return:
            d_result: np array (batch_size, n_input) - gradient with respect to input
        """
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0).reshape(self.B.value.shape)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        """
        Initializes the Convolution layer
        :param in_channels: int - number of input channels
        :param out_channels: int - number of output channels
        :param filter_size: int - size of the convolution filter
        :param padding: int - number of 'pixels' to pad on each side
        """
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding

    def forward(self, X):
        self.X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                        'constant', constant_values=0)
        batch_size, height, width, channels = self.X.shape
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                result[:, x, y, :] = (self.X[:, x:x+self.filter_size, y:y+self.filter_size, :].reshape(batch_size, -1) @
                                      self.W.value.reshape(-1, self.out_channels))
        result += self.B.value.reshape((1, 1, 1, -1))
        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_in = np.zeros((batch_size, height, width, channels))
        self.B.grad += np.sum(d_out.reshape(-1, out_channels), axis=0)
        for y in range(out_height):
            for x in range(out_width):

                self.W.grad += (self.X[:, x:x+self.filter_size, y:y+self.filter_size, :].reshape(batch_size, -1).T @
                                d_out[:, x, y, :].reshape(batch_size, out_channels)).reshape(self.W.grad.shape)
                d_in[:, x:x+self.filter_size, y:y+self.filter_size, :] += (d_out[:, x, y, :].reshape(batch_size, out_channels) @
                                                                           self.W.value.reshape(-1, out_channels).T).reshape(batch_size, self.filter_size, self.filter_size, channels)

        return d_in[:, self.padding:height-self.padding, self.padding:width-self.padding, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        """
        Initializes the max pool
        :param pool_size: int - area to pool
        :param stride: int - step size between pooling windows
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.chosen = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = self.X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        result = np.zeros((batch_size, out_height, out_width, channels))
        self.chosen = np.zeros_like(self.X)
        for x in range(out_height):
            for y in range(out_width):
                perception = X[:, x*self.stride:x*self.stride + self.pool_size,
                               y*self.stride:y*self.stride + self.pool_size, :]
                result[:, x, y, :] = np.max(perception, axis=(1, 2))
                ind = np.argmax(perception.reshape((batch_size, -1, channels)), axis=1)
                remember = np.zeros_like(perception).reshape((batch_size, -1, channels))
                remember[np.repeat(np.arange(0, batch_size), channels),
                         ind.ravel(), np.tile(np.arange(0, channels), batch_size)] += 1
                self.chosen[:, x*self.stride:x*self.stride + self.pool_size,
                            y*self.stride:y*self.stride + self.pool_size, :] = remember.reshape((batch_size,
                                                                                                 self.pool_size,
                                                                                                 self.pool_size,
                                                                                                 channels))
        return result

    def backward(self, d_out):
        return self.chosen * d_out.repeat(self.pool_size, axis=1).repeat(self.pool_size, axis=2)

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(self.X_shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
