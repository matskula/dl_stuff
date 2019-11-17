import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> ReLU -> Maxpool[4x4] ->
    Conv[3x3] -> ReLU -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        :param input_shape: tuple of 3 ints - image_width, image_height, n_channels
        :param n_output_classes: int - number of classes to predict
        :param conv1_channels: int - number of filters in the 1st conv layer
        :param conv2_channels: int - number of filters in the 2nd conv layer
        """
        self.convolution_one = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.relu_one = ReLULayer()
        self.maxpool_one = MaxPoolingLayer(4, 4)
        self.convolution_two = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu_two = ReLULayer()
        self.maxpool_two = MaxPoolingLayer(4, 4)
        self.flattener = Flattener()
        height = ((input_shape[0] + 2*1 - 3 + 1) // 4 + 2*1 - 3 + 1) // 4
        width = ((input_shape[1] + 2*1 - 3 + 1) // 4 + 2*1 - 3 + 1) // 4
        self.fc = FullyConnectedLayer(width*height*conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients on a batch of training examples
        :param X: np array (batch_size, height, width, input_features) - input data
        :param y: np array of int (batch_size) - classes
        :return: cross-entropy loss using soft-max
        """
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        loss, grad = softmax_with_cross_entropy(
            self.fc.forward(
                self.flattener.forward(
                    self.maxpool_two.forward(
                        self.relu_two.forward(
                            self.convolution_two.forward(
                                self.maxpool_one.forward(
                                    self.relu_one.forward(
                                        self.convolution_one.forward(X)))))))), y)
        self.convolution_one.backward(
            self.relu_one.backward(
                self.maxpool_one.backward(
                    self.convolution_two.backward(
                        self.relu_two.backward(
                            self.maxpool_two.backward(
                                self.flattener.backward(
                                    self.fc.backward(grad))))))))
        return loss

    def predict(self, X):
        preds = self.fc.forward(
                self.flattener.forward(
                    self.maxpool_two.forward(
                        self.relu_two.forward(
                            self.convolution_two.forward(
                                self.maxpool_one.forward(
                                    self.relu_one.forward(
                                        self.convolution_one.forward(X))))))))
        y_pred = np.argmax(preds, axis=1)
        return y_pred

    def params(self):
        result = {
            'conv1_W': self.convolution_one.W,
            'conv1_B': self.convolution_one.B,
            'conv2_W': self.convolution_two.W,
            'conv2_B': self.convolution_two.B,
            'fc_W': self.fc.W,
            'fc_B': self.fc.B
        }
        return result
