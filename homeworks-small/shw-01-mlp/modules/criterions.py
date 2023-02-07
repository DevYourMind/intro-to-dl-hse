import numpy as np
import scipy.special as sps
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, f"input and target shapes not matching: {input.shape = }; {target.shape = }"
        return ((input - target) ** 2).mean()

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, f"input and target shapes not matching {input.shape = }; {target.shape = }"
        n = input.size
        return 2 * (input - target) / n


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """

    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        return -sps.log_softmax(input, axis=1)[np.arange(len(target)), target].mean()

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        B = input.shape[0]
        ones = np.zeros_like(input)
        ones[np.arange(len(target)), target] = 1
        return (sps.softmax(input, axis=1) - ones) / B
