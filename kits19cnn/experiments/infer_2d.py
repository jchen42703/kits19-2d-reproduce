import torch

from kits19cnn.utils import softmax_helper
from kits19cnn.models import ResUNet, ResNetSeg
from kits19cnn.io import TestVoxelDataset
from .infer import BaseInferenceExperiment

class SegmentationInferenceExperiment2D(BaseInferenceExperiment):
    """
    Inference Experiment to support prediction experiments
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict):
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_datasets(self, test_ids):
        """
        Creates and returns the test dataset.
        """
        # creating the datasets
        test_dataset = TestVoxelDataset(im_ids=test_ids,
                                        file_ending=self.io_params["file_ending"])
        return test_dataset

    def get_model(self):
        model_name = self.model_params["model_name"]
        model_cls = globals()[model_name]
        model = model_cls(**self.model_params[model_name])
        model.inference_apply_nonlin = softmax_helper
        print(f"Using model, {model_name}")
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model
