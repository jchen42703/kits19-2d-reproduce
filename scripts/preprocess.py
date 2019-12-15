import os
from kits19cnn.io import Preprocessor

def main(config):
    """
    Main code for training a classification/seg/classification+seg model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. script/configs/stage1/preprocess.yml
    Returns:
        None
    """
    preprocess = Preprocessor(**config)
    if len(os.listdir(config["out_dir"])) == 0:
        print("Preprocessing the 3D volumes first...")
        preprocess.gen_data()

    print("Splitting the volumes into 2D numpy arrays...")
    preprocess.save_dir_as_2d()

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    with open(args.yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
