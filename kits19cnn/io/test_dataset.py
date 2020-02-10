from os.path import isfile, join
from glob import glob
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
    def __init__(self, im_ids: np.array, file_ending: str = ".npy",
                 load_labels: bool = False):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
        """
        super().__init__(im_ids=im_ids, file_ending=file_ending)
        self.load_labels = load_labels

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
            try:
                x = np.load(x_path)
                y = np.load(y_path) if self.load_labels else np.zeros(x.shape)
            except FileNotFoundError:
                x, y = self.load_all_slices_as_3d(case_id)
        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            x = nib.load(x_path).get_fdata()
            y = nib.load(y_path).get_fdata() if self.load_labels \
                else np.zeros(x.shape)
        return (x[None], y[None])

    def load_all_slices_as_3d(self, case_id):
        """
        Loads saved 2D numpy arrays (slices) into a single 3D volume. This
        will only work if self.file_ending is '.npy' and will work for both
        images and labels.

        Args:
            case_id (str): Path to the case folder.
        """
        x_slices = sorted(glob(join(case_id, "imaging_*.npy")))
        y_slices = sorted(glob(join(case_id, "segmentation_*.npy")))

        # BIG ASSUMPTION HERE VV (that both x and y have the same shape)
        slice_shape = np.load(x_slices[0]).shape # (256, 256) for stage1
        volume_shape = (len(x_slices),) + slice_shape
        x, y = np.zeros(volume_shape), np.zeros(volume_shape)
        for i, (x_path, y_path) in enumerate(zip(x_slices, y_slices)):
            x[i] = np.load(x_path)
            if self.load_labels:
                y[i] = np.load(y_path)

        return (x, y)
