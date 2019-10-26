import numpy as np


class SGD:
    def update(self, w, d_w, learning_rate):
        """
        Performs SGD update
        :param w: np array - weights
        :param d_w: np array, same shape as w - gradient
        :param learning_rate: float - learning rate
        :return:
            updated_weights, np array same shape as w
        """
        return w - d_w * learning_rate


class MomentumSGD:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = 0

    def update(self, w, d_w, learning_rate):
        """
        Performs Momentum SGD update
        :param w: np array - weights
        :param d_w: np array, same shape as w - gradient
        :param learning_rate: float - learning rate
        :return:
            updated_weights: np array same shape as w
        """
        self.velocity = self.velocity * self.momentum - learning_rate * d_w
        return w + self.velocity
