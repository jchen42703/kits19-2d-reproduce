from abc import abstractmethod
import torch

from kits19cnn.io import SliceDataset, PseudoSliceDataset
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

    def get_datasets(self, train_ids, valid_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        preprocess_t = get_preprocessing()
        # creating the datasets
        if self.io_params.get("pseudo_3D"):
            train_dataset = PseudoSliceDataset(im_ids=train_ids,
                                               in_dir=self.config["data_folder"],
                                               transforms=train_aug,
                                               preprocessing=preprocess_t,
                                               num_pseudo_slices=self.io_params["num_pseudo_slices"])
            valid_dataset = PseudoSliceDataset(im_ids=valid_ids,
                                               in_dir=self.config["data_folder"],
                                               transforms=val_aug,
                                               preprocessing=preprocess_t,
                                               num_pseudo_slices=self.io_params["num_pseudo_slices"])
        else:
            train_dataset = SliceDataset(im_ids=train_ids,
                                         in_dir=self.config["data_folder"],
                                         transforms=train_aug,
                                         preprocessing=preprocess_t)
            valid_dataset = SliceDataset(im_ids=valid_ids,
                                         in_dir=self.config["data_folder"],
                                         transforms=val_aug,
                                         preprocessing=preprocess_t)

        return (train_dataset, valid_dataset)

class TrainSegExperiment2D(TrainExperiment2D):
    """
    Stores the main parts of a segmentation experiment:
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
