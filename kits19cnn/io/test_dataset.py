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
    Same as VoxelDataset but also can load volumes by stacking slices.
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
        Returns:
            Tuple of:
            - pred (np.ndarray): shape (d, h, w)
            - y (np.ndarray): shape (d, h, w)
                if this does not exist, it's returned as a blank mask
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

class PredictionDataset(TestVoxelDataset):
    """
    For generating prediction data and masks. This is mainly used for the
    evaluation and ensembling (load_labels=False).
    """
    def __init__(self, in_dir: str, pred_dir: str, im_ids: np.array,
                 file_ending: str = ".npy", pred_prefix: str = "pred",
                 load_labels: bool = True):
        """
        Attributes
            in_dir (str): Path to the directory with the case folders with
                the labels
            pred_dir (str): Path to a prediction directory with case folders
                with predictions
            im_ids (np.ndarray): of case folder names. This is not the same
                as TestVoxelDataset, which assumes that `im_ids` is a numpy
                array of strings to FILEPATHS to case folders.
            file_ending (str): one of ['.npy', '.nii', '.nii.gz']
            pred_prefix (str): one of ['pred', 'pred_act']
                This is for flexibility for both ensembling and evaluation.
            load_labels (bool): whether or not to also load the label arrays.
                If False, the outputted label will just be an array of zeros
                with the same shape as the image. Defaults to True.
        """
        super().__init__(im_ids=im_ids, file_ending=file_ending,
                         load_labels=load_labels)
        self.in_dir = in_dir
        self.pred_dir = pred_dir
        self.pred_prefix = pred_prefix

    def load_volume(self, raw_case_id):
        """
        Loads volumes from either .npy or nifti files.
        Args:
            raw_case_id: raw case folder name
                i.e. case_00001
        Returns:
            Tuple of:
            - pred (np.ndarray): shape (channels, d, h, w)
            - y (np.ndarray): shape (1, d, h, w)
                if this does not exist, it's returned as a blank mask
        """
        pred_path = join(self.pred_dir, raw_case_id,
                         f"{self.pred_prefix}{self.file_ending}")
        y_path = join(self.in_dir, raw_case_id,
                      f"segmentation{self.file_ending}")
        if self.file_ending == ".npy":
            pred = np.load(pred_path)
            # vv makes sure that the prediction has the channels dimension
            pred = pred[None] if len(pred.shape) == 3 else pred

            if self.load_labels:
                try:
                    y = np.load(y_path)
                except FileNotFoundError:
                    y = self.load_all_label_slices_as_3d(raw_case_id)
            else:
                y = np.zeros(pred.shape)

        elif self.file_ending == ".nii.gz" or self.file_ending == ".nii":
            pred = nib.load(pred_path).get_fdata()
            y = nib.load(y_path).get_fdata() if self.load_labels \
                else np.zeros(pred.shape)

        return (pred, y[None])

    def load_all_label_slices_as_3d(self, raw_case_id):
        """
        Loads saved 2D numpy arrays (slices) into a single 3D volume. This
        will only work if self.file_ending is '.npy' and will work for both
        images and labels.

        Args:
            raw_case_id: raw case folder name
                i.e. case_00001
        Returns:
            y (np.ndarray): of shape (d, h, w)
        """
        y_slices = sorted(glob(join(self.in_dir, raw_case_id,
                                    "segmentation_*.npy")))

        slice_shape = np.load(y_slices[0]).shape # (256, 256) for stage1
        volume_shape = (len(y_slices),) + slice_shape
        y = np.zeros(volume_shape)

        for i, y_path in enumerate(y_slices):
            y[i] = np.load(y_path)

        return y
