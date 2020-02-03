import json
# import numpy as np
from tqdm import tqdm
import nibabel as nib
import numpy as np

from os.path import join
from pathlib import Path

from kits19cnn.utils import save_json

def gather_scale_ratio_dict(case_fpaths, resized_case_fpaths):
    """
    Gathers all (orig / resized) scale ratios in a dictionary
    Args:
        case_fpaths List[Path]: list of paths to case folders
    Returns:
        scale_ratio_dict (dict):
            Keys: case (raw case name, i.e. case_00000)
            Values: Shapes (tuple, i.e. (60, 500, 500))
    """
    scale_ratio_dict = {}
    for case, resized_case in tqdm(zip(case_fpaths, resized_case_fpaths),
                                   total=len(resized_case_fpaths)):
        orig_shape = np.array(nib.load(join(case, "imaging.nii.gz")).shape)
        resized_shape = np.array(np.load(join(resized_case, "imaging.npy")).shape)
        scale_ratio_dict[Path(case).name] = (orig_shape / resized_shape).tolist()
    return scale_ratio_dict

def save_scale_ratio_dict(case_fpaths, resized_case_fpaths, save_path):
    """
    Saves the dictionary of cases and scale ratios from `scale_ratio_dict`
    """
    scale_ratio_dict = gather_scale_ratio_dict(case_fpaths, resized_case_fpaths)
    save_json(scale_ratio_dict, save_path)
    return scale_ratio_dict
