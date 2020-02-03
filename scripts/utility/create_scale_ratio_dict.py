import os
from glob import glob
from kits19cnn.dataset_analyzer import save_scale_ratio_dict

def main(config):
    """
    Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. experiments/finetune_classification.yml
    Returns:
        None
    """
    search_path = os.path.join(config["in_dir"], "*/")
    resized_search_path = os.path.join(config["resized_in_dir"], "*/")
    case_list = sorted(glob(search_path))
    resized_case_list = sorted(glob(resized_search_path))
    save_scale_ratio_dict(case_list, resized_case_list,
                          config["scale_json_path"])

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
