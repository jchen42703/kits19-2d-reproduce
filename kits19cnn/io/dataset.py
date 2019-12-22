import os
from os.path import join
import numpy as np

import torch
from torch.utils.data import Dataset
from kits19cnn.io.preprocess import parse_slice_idx_to_str

class SliceDataset(Dataset):
    """
    Reads from a directory of 2D slice numpy arrays. Assumes the data
    directory contains 2D slices processed by
    `io.Preprocessor.save_dir_as_2d()`.

    B (background), K (kidney), KT (kidney + tumor)
    Stage 1: Sampled each class with p=0.33
    Stage 2: Samples only K and KT (p=0.5)
    """
    def __init__(self, im_ids: np.array, in_dir: str, transforms=None,
                 preprocessing=None):
        """
        Attributes
            im_ids (np.ndarray): of case_slice_idx_str.
            in_dir (str): path to where all of the cases and slices are located
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
        """
        print(f"Assuming inputs are .npy files...")
        self.im_ids = im_ids
        self.in_dir = in_dir
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        # loads data as a numpy arr and then adds the channel + batch size dimensions
        case_id = self.im_ids[idx]
        x, y = self.load_slices(case_id)
        x, y = self.apply_transforms_and_preprocessing(x, y)
        # conversion to tensor if needed
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
        # changing to channels first
        x, y = x.permute(2, 0, 1), y.permute(2, 0, 1)
        return (x, y)

    def __len__(self):
        return len(self.im_ids)

    def apply_transforms_and_preprocessing(x, y):
        """
        Function name ^, if applicable.
        Input arrays must be in channels_last format for albumentations
        to work properly.
        """
        if self.transforms:
            data_dict = self.transforms(image=x, mask=y)
            x, y = data_dict["image"], data_dict["mask"]

        if self.preprocessing:
            preprocessed = self.preprocessing(image=x, mask=y)
            x, y = preprocessed["image"], preprocessed["mask"]
        return (x, y)

    def load_slices(self, case_slice_idx_str):
        """
        Gets the slice idx using self.get_slice_idx_str() and actually loads
        the appropriate slice array.
        Loads the slices to the shape (h, w, 1).
        """
        case_fpath, slice_idx_str = self.split_case_slice_idx_str(case_slice_idx_str)
        x_path = join(case_fpath, f"imaging_{slice_idx_str}.npy")
        y_path = join(case_fpath, f"segmentation_{slice_idx_str}.npy")
        return (np.load(x_path)[:, :, None], np.load(y_path)[:, :, None])

    def split_case_slice_idx_str(self, case_slice_idx_str):
        """
        Processes the case_slice_idx_str (each element of self.im_ids)
        Returns:
            (full case path, parsed slice index in string form)
            Note: the parsed slice index is just the raw integer slice index
            with leading zeros until the total number of digits == 5.
        """
        case, slice_idx_str = case_slice_idx_str.split("_")
        case_fpath = join(self.in_dir, case)
        return (case_fpath, slice_idx_str)

class PseudoSliceDataset(SliceDataset):
    """
    Reads from a directory of 2D slice numpy arrays and samples positive
    slices. Assumes the data directory contains 2D slices processed by
    `io.Preprocessor.save_dir_as_2d()`. Generates 2.5D outputs

    B (background), K (kidney), KT (kidney + tumor)
    Stage 1: Sampled each class with p=0.33
    Stage 2: Samples only K and KT (p=0.5)
    """
    def __init__(self, im_ids: np.array, in_dir: str, transforms=None,
                 preprocessing=None, num_pseudo_slices=5):
        """
        Attributes
            im_ids (np.ndarray): of image names.
            in_dir (str): path to where all of the cases and slices are located
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
            num_pseudo_slices (int): number of pseudo 3D slices. Defaults to 1.
                1 meaning no pseudo slices. If it's greater than 1, it must
                be odd (even numbers above and below)
        """
        super().__init__(im_ids=im_ids, in_dir=in_dir,
                         transforms=transforms, preprocessing=preprocessing)
        self.num_pseudo_slices = num_pseudo_slices
        assert num_pseudo_slices % 2 == 1, \
            "`num_pseudo_slices` must be odd. i.e. 7 -> 3 above and 3 below"

    def load_slices(self, case_slice_idx_str):
        """
        Gets the slice idx using self.get_slice_idx_str() and actually loads
        the appropriate slice array. Returned arrays have shape:
            (batch_size, n_channels, h, w)
        for batchgenerators transforms.
        """
        case_fpath, center_slice_idx_str = self.split_case_slice_idx_str(case_slice_idx_str)
        center_slice_idx = int(center_slice_idx_str)
        min = center_slice_idx - (self.num_pseudo_slices - 1) // 2
        max = center_slice_idx + (self.num_pseudo_slices - 1) // 2 + 1

        x_path = join(case_fpath, f"imaging_{center_slice_idx_str}.npy")
        y_path = join(case_fpath, f"segmentation_{center_slice_idx_str}.npy")
        center_x, center_y = np.load(x_path)[:, :, None], np.load(y_path)[:, :, None]

        if self.num_pseudo_slices == 1:
            return (center_x, center_y)
        elif self.num_pseudo_slices > 1:
            # total shape: (h, w, num_pseudo_slices)
            x_arr = np.zeros(center_x.shape[:-1], (self.num_pseudo_slices,))
            for idx, slice_idx in enumerate(range(min, max)):
                slice_idx_str = parse_slice_idx_to_str(slice_idx)
                x_path = join(case_fpath, f"imaging_{slice_idx_str}.npy")
                # loading slices if they exist
                if os.path.isfile(x_path):
                    x_arr[:, :, idx] = np.load(x_path)
                else:
                    raise OSError(f"{x_path} not found.")
            return (x_arr, center_y)
