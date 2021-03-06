from pathlib import Path
from abc import abstractmethod
from os.path import join, isdir
from tqdm import tqdm
import os
import numpy as np
import inspect
import nibabel as nib
import torch

class BasePredictor(object):
    """
    Inference for a single model for every file generated by `test_loader`.
    Predictions are saved in `out_dir`.
    """
    def __init__(self, out_dir, model, test_loader):
        """
        Attributes
            out_dir (str): path to the output directory to store predictions
            model (torch.nn.Module): single model
            test_loader: Iterable instance for generating data
                (pref. torch DataLoader)
        """
        self.model = model
        self.out_dir = out_dir
        if not isdir(self.out_dir):
            os.mkdir(self.out_dir)
            print(f"Created {self.out_dir}!")
        self.test_loader = test_loader

    @abstractmethod
    def run_3D_predictions(self):
        """
        Runs predictions on the dataset (specified in test_loader)
        """
        return

    def save_pred(self, pred, act, case):
        """
        Saves both prediction and activation maps in `out_dir` in the
        KiTS19 format.
        Args:
            pred (np.ndarray): shape (x, y, z)
            act (np.ndarray): shape (n_classes, x, y, z)
            case: path to a case folder (an element of self.cases)
        Returns:
            None
        """
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        np.save(join(out_case_dir, "pred.npy"), pred)
        np.save(join(out_case_dir, "pred_act.npy"), act)
