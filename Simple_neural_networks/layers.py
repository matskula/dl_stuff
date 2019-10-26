import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    :param W: np array - weights
    :param reg_strength: float value
    :return:
        loss: single value - l2 regularization loss
        gradient: np.array same shape as W - gradient of weight by l2 loss
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
