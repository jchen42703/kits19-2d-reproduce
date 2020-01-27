from os.path import isfile, join
import numpy as np

import torch
from torch.utils.data import Dataset

class VoxelDataset(Dataset):
    def __init__(self, im_ids: np.array, file_ending: str = ".npy"):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
        """
        self.im_ids = im_ids
        self.file_ending = file_ending
        print(f"Using the {file_ending} files...")

    def __getitem__(self, idx):
        # loads data as a numpy arr and then adds the channel + batch size dimensions
        case_id = self.im_ids[idx]
        x, y = self.load_volume(case_id)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.im_ids)

    def load_volume(self, case_id):
        """
        Loads volume from either .npy or nifti files.
        Args:
            case_id: path to the case folder
                i.e. /content/kits19/data/case_00001
        Returns:
            Tuple of:
            - x (np.ndarray): shape (1, d, h, w)
            - y (np.ndarray): same shape as x
        """
        x_path = join(case_id, f"imaging{self.file_ending}")
        y_path = join(case_id, f"segmentation{self.file_ending}")
        if self.file_ending == ".npy":
            x, y = np.load(x_path), np.load(y_path)
        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            x, y = nib.load(x_path).get_fdata(), nib.load(y_path).get_fdata()
        return (x[None], y[None])

class TestVoxelDataset(VoxelDataset):
    """
    Same as VoxelDataset, but can handle when there are no masks (just returns
    blank masks). This is a separate class to prevent lowkey errors with
    blank masks--VoxelDataset explicitly fails when there are no masks.

    Make sure to specify the correct input directory in the `im_ids` for the
    volumes to load correctly.
    """
    def __init__(self, im_ids: np.array, file_ending: str = ".npy"):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
        """
        super().__init__(im_ids=im_ids, file_ending=file_ending)

    def load_volume(self, case_id):
        """
        Loads volume from either .npy or nifti files.
        Args:
            case_id: path to the case folder
                i.e. /content/kits19/data/case_00001
        Returns:
            Tuple of:
            - x (np.ndarray): shape (1, d, h, w)
            - y (np.ndarray): same shape as x
                if this does not exist, it's returned as a blank mask
        """
        x_path = join(case_id, f"imaging{self.file_ending}")
        y_path = join(case_id, f"segmentation{self.file_ending}")
        if self.file_ending == ".npy":
            x = np.load(x_path)
            y = np.load(y_path) if isfile(y_path) else np.zeros(x.shape)
        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            x = nib.load(x_path).get_fdata()
            y = nib.load(y_path).get_fdata() if isfile(y_path) else np.zeros(x.shape)
        return (x[None], y[None])
