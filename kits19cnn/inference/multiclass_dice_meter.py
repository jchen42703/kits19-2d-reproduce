import numpy as np

from catalyst.utils.meters import meter
from catalyst.utils.confusion_matrix import (
    calculate_confusion_matrix_from_arrays, calculate_tp_fp_fn
)
from catalyst.dl.callbacks.metrics.functional import calculate_dice


class MultiClassDiceMeter(meter.Meter):
    """
    Keeps track of global true positives, false positives, and false negatives
    for each epoch and calculates multi-class F1-score based on
    those metrics.
    """
    def __init__(self, num_classes=3):
        super(MultiClassDiceMeter, self).__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """
        Resets the confusion matrix holding the tp/tn/fp/fn to 0.
        """
        self.confusion_matrix = None

    def add(self, output, target):
        """
        Records the confusion matrix for the array.

        Args:
            output (np.ndarray):
                prediction after activation function
                shape should be (batch_size, ...), but works with any shape

                If you are using softmax and argmax, then the shape should be
                (batch_size, num_classes, ...)
            target (np.ndarray):
                If you are using softmax and argmax, then the shape should be
                (batch_size, 1, ...)
        Returns:
            None
        """
        output = output.argmax(1)

        confusion_matrix = calculate_confusion_matrix_from_arrays(
            target, output, self.num_classes
        )

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def value(self):
        """
        Calculates the dice epoch-wise using the running confusion matrix.

        Args:
            None
        Returns:
            dice_scores (np.ndarray): of class-wise dice scores
        """
        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        dice_scores: np.ndarray = calculate_dice(**tp_fp_fn_dict)

        return dice_scores
