from tqdm import tqdm
import os
from os.path import join
import nibabel as nib

from kits19cnn.io import get_bbox_from_mask, expand_bbox
from kits19cnn.utils import save_json

def create_bbox_stage1(mask):
    """
    Creates the bounding box from mask and expands it to 256 by 464. DOES NOT
    RESIZE THE BBOX LIKE IN STAGE 1.
    Args:
        mask (np.ndarray): 3D Array (no channels)
    Returns:
        expanded_bbox (list):
            [[lb_x, ub_x], [lb_y, ub_y], [lb_z, ub_z]]
            where lb -> lower bound coordinate
                  ub -> upper bound coordinate
    """
    bbox = get_bbox_from_mask(mask, outside_value=0)
    expanded_bbox = expand_bbox(bbox,
                                bbox_lengths=[None, 256, 464])
                                # Changed to 256, 464 because the max y length
                                # in the uninterpolated dset is 459 and it
                                # needs to be divisible by 16
    return expanded_bbox

def fetch_cases(in_dir):
    """
    Creates a list of all available case folders in `in_dir`
    """
    cases_raw = [case \
                 for case in os.listdir(in_dir) \
                 if case.startswith("case")]
    cases_raw = sorted(cases_raw)
    assert len(cases_raw) > 0, \
        "Please make sure that in_dir refers to the proper directory."
    return cases_raw[:210] # past 210 are the test cases with no masks

def main(cases_raw, in_dir, save_path):
    """
    Reads all raw masks and creates bbox. This bbox is then expanded to 256 by
    256 (z-dim stays the same).
    """
    cases_raw = cases_raw if cases_raw is not None else fetch_cases(in_dir)
    actual_bbox_dict = {}
    for case_raw in tqdm(cases_raw):
        mask = nib.load(join(in_dir, case_raw, "segmentation.nii.gz")).get_fdata()
        actual_bbox_dict[case_raw] = create_bbox_stage1(mask)
    save_json(actual_bbox_dict, save_path)

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="For prediction.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    with open(args.yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(**config)
