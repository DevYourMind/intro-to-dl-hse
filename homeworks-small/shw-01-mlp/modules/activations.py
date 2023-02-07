import numpy as np
from .base import Module
import scipy.special as sps


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return grad_output * (input > 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return sps.expit(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        sigmoid = sps.expit(input)
        return grad_output * sigmoid * (1 - sigmoid)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return sps.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        softmax = sps.softmax(input, axis=1)
        mult = grad_output * softmax
        return -mult.sum(axis=1).reshape(-1, 1) * softmax + mult


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        return sps.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
        softmax = sps.softmax(input, axis=1)
        return -grad_output.sum(axis=1).reshape(-1, 1) * softmax + grad_output
