from kits19cnn.experiments import SegmentationInferenceExperiment2D, \
                                  seed_everything
import os
import shutil
from tqdm import tqdm
from pathlib import Path

def copy_files(img_fpaths, out_dir):
    for fpath in tqdm(img_fpaths):
        base_name = Path(fpath).name
        shutil.copytree(fpath, os.path.join(out_dir, base_name))

def main(config, out_path):
    """
    Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. experiments/finetune_classification.yml
    Returns:
        None
    """
    # setting up the train/val split with filenames
    seed = config["io_params"]["split_seed"]
    seed_everything(seed)
    exp = SegmentationInferenceExperiment2D(config)
    print(f"Seed: {seed}")
    test_ids = exp.test_dset.im_ids
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        print(f"Created {out_path}")
    print(f"Copying {len(test_ids)} test files to a {out_path}")
    copy_files(test_ids, out_path)

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

    main(config)
