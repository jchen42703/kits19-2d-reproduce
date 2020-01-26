import os
from os.path import join, isdir
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import nibabel as nib
import numpy as np
import json

from .resample import resample_patient
from .custom_augmentations import resize_data_and_seg, crop_to_bbox

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
                 bbox_json_path=None, clip_values=[-30, 300], with_mask=True,
                 fg_classes=[0, 1, 2], resize_xy_shape=(256, 256)):
        """
        Attributes:
            in_dir (str): directory with the input data. Should be the
                kits19/data directory.
            out_dir (str): output directory where you want to save each case
            cases: list of case folders to preprocess
            kits_json_path (str): path to the kits.json file in the kits19/data
                directory. This only should be specfied if you're resampling.
                Defaults to None.
            bbox_json_path (str): path to the bbox_stage1.json file made from
                stage1 post-processing. Triggers cropping to the bboxes.
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
        self._load_bbox_json(bbox_json_path)
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
        self.resize_xy_shape = tuple(resize_xy_shape)

    def gen_data(self, save_fnames=["imaging", "segmentation"]):
        """
        Generates and saves preprocessed data as numpy arrays (n, x, y).
        Args:
            task_path: file path to the task directory
                (must have the corresponding "dataset.json" in it)
            save_fnames (List[str]): save names for [image, seg] respectively.
                DOESN'T INCLUDE THE .npy
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
            if self.bbox_dict is not None:
                preprocessed_img, preprocessed_label = self.crop_case_to_bbox(preprocessed_img,
                                                                              preprocessed_label,
                                                                              case)
            self.save_imgs(preprocessed_img, preprocessed_label, case,
                           save_fnames=save_fnames)

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

    def save_imgs(self, image, mask, case,
                  save_fnames=["imaging", "segmentation"]):
        """
        Saves an image and mask pair as .npy arrays in the KiTS19 file structure
        Args:
            image: numpy array
            mask: numpy array
            case: path to a case folder (each element of self.cases)
            save_fnames (List[str]): save names for [image, seg] respectively.
                DOESN'T INCLUDE THE .npy
        """
        for fname in save_fnames:
            assert not ".npy" in fname, \
                "Filenames in save_fnames should not include .npy in the name."
        # saving the generated dataset
        # output dir in KiTS19 format
        # extracting the raw case folder name
        case_raw = Path(case).name # extracting the raw case folder name
        out_case_dir = join(self.out_dir, case_raw)
        # checking to make sure that the output directories exist
        if not isdir(out_case_dir):
            os.mkdir(out_case_dir)

        np.save(os.path.join(out_case_dir, f"{save_fnames[0]}.npy"), image)
        if mask is not None:
            np.save(os.path.join(out_case_dir, f"{save_fnames[1]}.npy"), mask)

    def save_dir_as_2d(self, base_fnames=["imaging", "segmentation"],
                       delete3dcase=False):
        """
        Takes preprocessed 3D numpy arrays and saves them as slices
        in the same directory.
        Arrays must have shape (n, h, w).

        Args:
            base_fnames (List[str]): names to read for [image, seg] respectively.
                DOESN'T INCLUDE THE .npy
            delete3dcase (bool): whether or not to delete the 3D volume after
                saving the 2D sliced versions
        """
        for fname in base_fnames:
            assert not ".npy" in fname, \
                "Filenames in base_fnames should not include .npy in the name."

        self.pos_per_class_dict = {} # saves slices per class
        self.pos_per_slice_dict = defaultdict(list) # saves classes per slice
        # Generating data and saving them recursively
        for case in tqdm(self.cases):
            # output dir in KiTS19 format
            case_raw = Path(case).name # extracting the raw case folder name
            out_case_dir = join(self.out_dir, case_raw)
            # checking to make sure that the output directories exist
            if not isdir(out_case_dir):
                os.mkdir(out_case_dir)
            # assumes the .npy files have shape: (d, h, w)
            paths = [join(out_case_dir, f"{base_fnames[0]}.npy"),
                     join(out_case_dir, f"{base_fnames[1]}.npy")]
            image, label = np.load(paths[0]), np.load(paths[1])
            self.save_3d_as_2d(image, label, case_raw, out_case_dir)

            # to deal with colaboratory storage limitations
            if delete3dcase:
                os.remove(paths[0]), os.remove(paths[1])

        if self.fg_classes is not None:
            self._save_pos_slice_dict()

    def save_3d_as_2d(self, image, mask, case_raw, out_case_dir):
        """
        Saves a 3D volume as separate 2D arrays for each slice across the
        axial axis. The naming convention is as follows:
            imaging_{parsed_slice_idx}.npy
            segmentation_{parsed_slice_idx}.npy
            where parsed_slice_idx is just the slice index but filled with
            zeros until it hits 5 digits (so sorting is easier.)
        Args:
            image: numpy array
            mask: numpy array
            case: raw case folder name
        """
        # saving the generated dataset
        # iterates through all slices and saves them individually as 2D arrays
        assert len(image.shape) == 3, \
            "Image shape should be (n, h, w)"

        slice_idx_per_class = defaultdict(list)
        for slice_idx in range(image.shape[0]):
            # naming
            slice_idx_str = parse_slice_idx_to_str(slice_idx)
            case_str = f"{case_raw}_{slice_idx_str}"

            if mask is not None:
                label_slice = mask[slice_idx]
            # appending fg slice indices
            if self.fg_classes is not None:
                for label_idx in self.fg_classes:
                    if label_idx != 0 and (label_slice == label_idx).any():
                        slice_idx_per_class[label_idx].append(slice_idx)
                        self.pos_per_slice_dict[case_str].append(label_idx)
                    elif label_idx == 0 and np.sum(label_slice) == 0:
                        # for completely blank labels
                        slice_idx_per_class[label_idx].append(slice_idx)
                        self.pos_per_slice_dict[case_str].append(label_idx)

            self._save_slices(image, mask, out_case_dir=out_case_dir,
                              slice_idx=slice_idx, slice_idx_str=slice_idx_str)

        if self.fg_classes is not None:
            self.pos_per_class_dict[case_raw] = slice_idx_per_class

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
            - classes_per_slice.json
                the keys are not cases, but the actual filenames that are
                being read.
                    {
                        case_slice_idx_str: [classes_in_slice],
                        case_slice_idx_str2: [classes_in_slice],
                    }
        """
        save_path_per_slice = join(self.out_dir, "classes_per_slice.json")
        # saving the dictionaries
        print(f"Logged the classes in {self.fg_classes} for each slice at",
              f"{save_path_per_slice}.")
        with open(save_path_per_slice, "w") as fp:
            json.dump(self.pos_per_slice_dict, fp)

        save_path = join(self.out_dir, "slice_indices.json")
        # saving the dictionaries
        print(f"Logged the slice indices for each class in {self.fg_classes} at",
              f"{save_path}.")
        with open(save_path, "w") as fp:
            json.dump(self.pos_per_class_dict, fp)

    def _save_slices(self, image, mask, out_case_dir, slice_idx,
                     slice_idx_str):
        """
        For saving the slices in self.save_3d_as_2d()
        """
        np.save(join(out_case_dir, f"imaging_{slice_idx_str}.npy"),
                image[slice_idx])
        if mask is not None:
            label_slice = mask[slice_idx]
            np.save(join(out_case_dir, f"segmentation_{slice_idx_str}.npy"),
                    label_slice)

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

    def _load_bbox_json(self, json_path):
        """
        Loads the kits.json file into `self.kits_json`
        """
        if json_path is None:
            self.bbox_dict = None
            print("bbox_json_path, so not cropping volumes to their bbox.")
        else:
            with open(json_path, "r") as fp:
                self.bbox_dict = json.load(fp)

    def crop_case_to_bbox(self, image, label, case):
        """
        Crops a 3D image and 3D label to the corresponding bounding box.
        """
        bbox_coord = self.bbox_dict[case]
        return (crop_to_bbox(image, bbox), crop_to_bbox(label, case))

def standardize_per_image(image):
    """
    Z-score standardization per image.
    """
    mean, stddev = image.mean(), image.std()
    return (image - mean) / stddev

def parse_slice_idx_to_str(slice_idx):
    """
    Parse the slice index to a three digit string for saving and reading the
    2D .npy files generated by io.preprocess.Preprocessor.

    Naming convention: {type of slice}_{case}_{slice_idx}
        * adding 0s to slice_idx until it reaches 3 digits,
        * so sorting files is easier when stacking
    """
    return f"{slice_idx:03}"
