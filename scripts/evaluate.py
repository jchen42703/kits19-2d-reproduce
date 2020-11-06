from kits19cnn.inference import GlobalMetricsEvaluator

def main(config):
    """
    Main code for running the evaluation of 3D volumes.
    Args:
        config (dict): dictionary read from a yaml file
            i.e. script_configs/eval.yml
    Returns:
        None
    """
    evaluator = GlobalMetricsEvaluator(config["orig_img_dir"],
                                       config["pred_dir"],
                                       label_file_ending=config["label_file_ending"],
                                       binary_tumor=config["binary_tumor"],
                                       num_classes=config["num_classes"])
    evaluator.evaluate_all(print_metrics=config["print_metrics"])

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="For evaluation.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    with open(args.yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
