import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        :param n_input: int - dimension of the model input
        :param n_output: int - number of classes to predict
        :param hidden_layer_size: int - number of neurons in the hidden layer
        :param reg: float - L2 regularization strength
        """
        self.reg = reg
        self.layer_one = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer_ReLU = ReLULayer()
        self.layer_two = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        :param X: np array (batch_size, input_features) - input data
        :param y: np array of int (batch_size) - classes
        :return:
            loss: float - single value, cross entropy loss
        """
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)

        loss, grad = softmax_with_cross_entropy(
            self.layer_two.forward(
                self.layer_ReLU.forward(
                    self.layer_one.forward(X))), y)
        self.layer_one.backward(
            self.layer_ReLU.backward(
                self.layer_two.backward(grad)))
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        :param X: np array (test_samples, num_features)
        :return:
            y_pred: np.array of int (test_samples)
        """
        preds = self.layer_two.forward(
                    self.layer_ReLU.forward(
                        self.layer_one.forward(X)))

        y_pred = np.argmax(preds, axis=1)
        return y_pred

    def params(self):
        result = {'layer_one_W': self.layer_one.W, 'layer_one_B': self.layer_one.B,
                  'layer_two_W': self.layer_two.W, 'layer_two_B': self.layer_two.B}
        return result
