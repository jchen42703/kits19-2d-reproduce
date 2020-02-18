from abc import abstractmethod
import torch

from kits19cnn.io import SliceDataset, SliceDatasetOnTheFly, \
                         PseudoSliceDataset, PseudoSliceDatasetOnTheFly
from kits19cnn.models import ResUNet, ResNetSeg
from .utils import get_training_augmentation, get_validation_augmentation, \
                   get_preprocessing
from .train import TrainExperiment

class TrainExperiment2D(TrainExperiment):
    """
    Stores the main parts of a experiment with 2D images:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    @abstractmethod
    def get_model(self):
        """
        Creates and returns the model.
        """
        return

    def get_datasets(self, train_ids, val_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        preprocess_t = get_preprocessing()

        train_kwargs = {"im_ids": train_ids,
                        "in_dir": self.config["data_folder"],
                        "transforms": train_aug,
                        "preprocessing": preprocess_t}
        val_kwargs = {"im_ids": val_ids,
                      "in_dir": self.config["data_folder"],
                      "transforms": val_aug,
                      "preprocessing": preprocess_t}

        dset_cls_name = "SliceDataset"
        # updating kwargs and dynamically fetching the dataset class name
        if self.io_params.get("pseudo_3D"):
            num_pseudo_slices = self.io_params["num_pseudo_slices"]
            train_kwargs.update({"num_pseudo_slices": num_pseudo_slices})
            val_kwargs.update({"num_pseudo_slices": num_pseudo_slices})
            dset_cls_name = "PseudoSliceDataset"
        if self.io_params["sample_on_the_fly"]:
            sample_distr = self.io_params["sampling_distribution"]
            train_kwargs.update({"sampling_distribution": sample_distr,
                                 "pos_slice_dict": self._pos_slice_dict})
            val_kwargs.update({"sampling_distribution": sample_distr,
                               "pos_slice_dict": self._pos_slice_dict})
            dset_cls_name = f"{dset_cls_name}OnTheFly"

        # instantiating the dataset classes using the dset_cls_name
        train_dataset = globals()[dset_cls_name](**train_kwargs)
        val_dataset = globals()[dset_cls_name](**val_kwargs)

        return (train_dataset, val_dataset)

class TrainSegExperiment2D(TrainExperiment2D):
    """
    Stores the main parts of a segmentation experiment:
    - dataset split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_model(self):
        model_name = self.model_params["model_name"]
        model_cls = globals()[model_name]
        model = model_cls(**self.model_params[model_name])
        print(f"Using model, {model_name}")
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model
