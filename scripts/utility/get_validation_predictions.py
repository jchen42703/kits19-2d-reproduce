from kits19cnn.inference import General3DPredictor
from kits19cnn.experiments import SegmentationInferenceExperiment2D, \
                                  seed_everything

def main(config):
    """
    Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
    Returns:
        None
    """
    # setting up the train/val split with filenames
    seed = config["io_params"]["split_seed"]
    seed_everything(seed)
    exp = SegmentationInferenceExperiment2D(config)
    # Changing to validation
    print("\nChanging to validation...")
    val_ids = exp.get_split()[1]
    print(f"Inferring on {len(val_ids)} validation cases")
    exp.test_dset = exp.get_datasets(val_ids)
    exp.loaders = exp.get_loaders()

    print(f"Seed: {seed}")
    pred = General3DPredictor(out_dir=config["out_dir"],
                              model=exp.model, test_loader=exp.loaders["test"],
                              pred_3D_params=config["predict_3D_params"],
                              pseudo_3D=config.get("pseudo_3D"))
    pred.run_3D_predictions()

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
