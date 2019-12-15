import os
from os.path import join, isdir
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import nibabel as nib
import numpy as np
import json

from kits19cnn.io.resample import resample_patient
from kits19cnn.io.custom_augmentations import resize_data_and_seg

class Preprocessor(object):
    """
    Preprocesses the original dataset (interpolated).
    Procedures:
        * Resampled all volumes to have a thickness of 3mm.
        * Clipped to [-30, 300] HU
        * z-score standardization (zero mean and unit variance)
            * Standardization per 3D image instead of ACROSS THE WHOLE
            TRAINING SET
        * save as .npy array
            * imaging.npy
            * segmentation.npy (if with_masks)
    """
    def __init__(self, in_dir, out_dir, cases=None, kits_json_path=None,
                 clip_values=[-30, 300], with_mask=True, fg_classes=[0, 1, 2],
                 resize_xy_shape=(256, 256)):
        """
        Attributes:
            in_dir (str): directory with the input data. Should be the
                kits19/data directory.
            out_dir (str): output directory where you want to save each case
            cases: list of case folders to preprocess
            kits_json_path (str): path to the kits.json file in the kits19/data
                directory. This only should be specfied if you're resampling.
                Defaults to None.
            target_spacing (list/tuple): spacing to resample to
            clip_values (list, tuple): values you want to clip CT scans to.
                Defaults to None for no clipping.
            with_mask (bool): whether or not to preprocess with masks or no
                masks. Applicable to preprocessing test set (no labels
                available).
            fg_classes (list): of foreground class indices
                if None, doesn't gather fg class stats.
        """
        self.in_dir = in_dir
        self.out_dir = out_dir

        self._load_kits_json(kits_json_path)
        self.clip_values = clip_values
        self.with_mask = with_mask
        self.fg_classes = fg_classes
        if not self.with_mask:
            assert self.fg_classes is None, \
                "When with_mask is False, fg_classes must be None."
        self.cases = cases
        # automatically collecting all of the case folder names
        if self.cases is None:
            self.cases = [os.path.join(self.in_dir, case) \
                          for case in os.listdir(self.in_dir) \
                          if case.startswith("case")]
            self.cases = sorted(self.cases)
            assert len(self.cases) > 0, \
                "Please make sure that in_dir refers to the proper directory."
        # making directory if out_dir doesn't exist
        if not isdir(out_dir):
            os.mkdir(out_dir)
            print("Created directory: {0}".format(out_dir))
        self.resize_xy_shape = resize_xy_shape

    def gen_data(self):
        """
        Generates and saves preprocessed data as numpy arrays (n, x, y).
        Args:
            task_path: file path to the task directory
                (must have the corresponding "dataset.json" in it)
        Returns:
            None
        """
        # Generating data and saving them recursively
        for case in tqdm(self.cases):
            x_path, y_path = join(case, "imaging.nii.gz"), join(case, "segmentation.nii.gz")
            image = nib.load(x_path).get_fdata()[None]
            label = nib.load(y_path).get_fdata()[None] if self.with_mask \
                    else None
            preprocessed_img, preprocessed_label = self.preprocess(image,
                                                                   label,
                                                                   case)

            self.save_imgs(preprocessed_img, preprocessed_label, case)

    def preprocess(self, image, mask, case=None):
        """
        Clipping, cropping, and resampling.
        Args:
            image: numpy array
                shape (c, n, x, y)
            mask: numpy array or None
                shape (c, n, x, y)
            case (str): path to a case folder
        Returns:
            tuple of:
                - preprocessed image
                    shape: (n, x, y)
                - preprocessed mask or None
                    shape: (n, x, y)
        """
        raw_case = Path(case).name # raw case name, i.e. case_00000
        # resampling
        if self.kits_json is not None:
            for info_dict in self.kits_json:
                # guaranteeing that the info is corresponding to the right
                # case
                if info_dict["case_id"] == raw_case:
                    case_info_dict = info_dict
                    break
            # resampling the slices axis to 3mm
            orig_spacing = (case_info_dict["captured_slice_thickness"],
                            case_info_dict["captured_pixel_width"],
                            case_info_dict["captured_pixel_width"])
            target_spacing = (3,) + orig_spacing[1:]
            image, mask = resample_patient(image, mask, np.array(orig_spacing),
                                           target_spacing=np.array(target_spacing))
        if self.clip_values is not None:
            image = np.clip(image, self.clip_values[0], self.clip_values[1])

        if self.resize_xy_shape is not None:
            # image coming in : shape (c, n, h, w); mask is same shape
            zdim_size = image.shape[1]
            resize_xy_shape = (zdim_size,) + self.resize_xy_shape
            image, mask = resize_data_and_seg(image, size=resize_xy_shape,
                                              seg=mask)
        image = standardize_per_image(image)
        mask = mask.squeeze() if mask is not None else mask
        return (image.squeeze(), mask)

    def save_imgs(self, image, mask, case):
        """
        Saves an image and mask pair as .npy arrays in the KiTS19 file structure
        Args:
            image: numpy array
            mask: numpy array
            case: path to a case folder (each element of self.cases)
        """
        # saving the generated dataset
        # output dir in KiTS19 format
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        np.save(os.path.join(out_case_dir, "imaging.npy"), image)
        if mask is not None:
            np.save(os.path.join(out_case_dir, "segmentation.npy"), mask)

    def save_dir_as_2d(self):
        """
        Takes preprocessed 3D numpy arrays and saves them as slices
        in the same directory.
        Arrays must have shape (n, h, w).
        """
        self.pos_slice_dict = {}
        # Generating data and saving them recursively
        for case in tqdm(self.cases):
            # assumes the .npy files have shape: (d, h, w)
            image = np.load(join(case, "imaging.npy"))
            label = np.load(join(case, "segmentation.npy"))
            self.save_3d_as_2d(image, label, case)
        if self.fg_classes is not None:
            self._save_pos_slice_dict()

    def save_3d_as_2d(self, image, mask, case):
        """
        Saves an image and mask pair as .npy arrays in the
        KiTS19 file structure
        Args:
            image: numpy array
            mask: numpy array
            case: path to a case folder (each element of self.cases)
        """
        # saving the generated dataset
        # output dir in KiTS19 format
        # extracting the raw case folder name
        case = Path(case).name
        out_case_dir = join(self.out_dir, case)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        # iterates through all slices and saves them individually as 2D arrays
        fg_indices = defaultdict(list)
        assert len(image.shape) == 3, \
            "Image shape should be (n, h, w)"

        for slice_idx in range(image.shape[0]):
            # appending fg slice indices
            if mask is not None:
                label_slice = mask[slice_idx]

            for idx in self.fg_classes:
                if idx != 0 and (label_slice == idx).any():
                    fg_indices[idx].append(slice_idx)
                elif idx == 0 and np.sum(label_slice) == 0:
                    # for completely blank labels
                    fg_indices[idx].append(slice_idx)

            slice_idx_str = parse_slice_idx_to_str(slice_idx)
            np.save(join(out_case_dir, f"imaging_{slice_idx_str}.npy"),
                    image[slice_idx])
            if mask is not None:
                np.save(join(out_case_dir, f"segmentation_{slice_idx_str}.npy"),
                        label_slice)
        # {case1: [idx1, idx2,...], case2: ...}
        self.pos_slice_dict[case] = fg_indices

    def _save_pos_slice_dict(self):
        """
        Saves the foreground (positive) class dictionaries:
            - slice_indices.json
                saves the slice indices per class
                    {
                        case: {fg_class1: [slice indices...],
                               fg_class2: [slice indices...],
                               ...}
                    }
            - slice_indices_general.json
                saves the slice indices for all foreground classes into a
                    single list
                    {case: [slice indices...],}
        """
        save_path = join(self.out_dir, "slice_indices.json")
        # saving the dictionaries
        print(f"Logged the slice indices for each class in {self.fg_classes} at"
              f"{save_path}.")
        with open(save_path, "w") as fp:
            json.dump(self.pos_slice_dict, fp)

    def _load_kits_json(self, json_path):
        """
        Loads the kits.json file into `self.kits_json`
        """
        if json_path is None:
            self.kits_json = None
            print("`kits_json_path is empty, so not resampling.`")
        elif json_path is not None:
            with open(json_path, "r") as fp:
                self.kits_json = json.load(fp)

def standardize_per_image(image):
    """
    Z-score standardization per image.
    """
    mean, stddev = image.mean(), image.std()
    return (image - mean) / stddev

def parse_slice_idx_to_str(self, slice_idx):
    """
    Parse the slice index to a three digit string for saving and reading the
    2D .npy files generated by io.preprocess.Preprocessor.

    Naming convention: {type of slice}_{case}_{slice_idx}
        * adding 0s to slice_idx until it reaches 3 digits,
        * so sorting files is easier when stacking
    """
    return f"{slice_idx:03}"
