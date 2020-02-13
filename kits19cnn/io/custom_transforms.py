from albumentations.core.transforms_interface import BasicTransform, \
                                                     DualTransform
import numpy as np
import torch

class ToTensorV2(BasicTransform):
    """
    Convert image and mask to `torch.Tensor`.
    """

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask.transpose(2, 0, 1))

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

class CenterCrop(DualTransform):
    """
    Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
        pad_kwargs (dict): kwargs for padding when the crop size is larger than
            the input image.
    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """

    def __init__(self, height, width, always_apply=False, p=1.0,
                 pad_kwargs={"mode": "constant", "constant_values": 0}):
        super(CenterCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.pad_kwargs = pad_kwargs

    def apply(self, img, **params):
        return center_crop(img, self.height, self.width,
                           pad_kwargs=self.pad_kwargs)

    def get_transform_init_args_names(self):
        return ("height", "width")

def center_crop(img, crop_height, crop_width, dim=2, pad_kwargs={}):
    """
    Center cropping 2D images (channels last)
    """
    data_shape, crop_size = img.shape, (crop_height, crop_width)
    # list of lower bounds for each axis
    lbs = get_lbs_for_center_crop(crop_size, data_shape)
    need_to_pad = [[abs(min(0, lbs[d])), abs(min(0, data_shape[d] - (lbs[d] + crop_size[d])))]
                    for d in range(dim)] + [[0, 0]]
    # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
    ubs = [min(lbs[d] + crop_size[d], data_shape[d]) for d in range(dim)]
    lbs = [max(0, lbs[d]) for d in range(dim)]

    slicer_data = [slice(0, data_shape[1])] + [slice(lbs[d], ubs[d])
                   for d in range(dim)]
    img_cropped = img[tuple(slicer_data)]
    img = np.pad(img_cropped, need_to_pad, **pad_kwargs)
    return img

def get_lbs_for_center_crop(crop_size, data_shape):
    """
    Fetches the lower bounds for central cropping.
    Args:
        crop_size: (height, width)
        data_shape: (x ,y, c)
    """
    lbs = []
    for i in range(len(data_shape)-1):
        lbs.append((data_shape[i] - crop_size[i]) // 2)
    return lbs
