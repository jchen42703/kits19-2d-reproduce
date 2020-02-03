import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation, \
                                                resize_multichannel_image

def resize_data_and_seg(data, size, seg=None, order_data=3,
                        order_seg=1, cval_seg=0):
    """
    Args:
        data (np.ndarray): shape (c, h, w (, d))
        seg (np.ndarray): shape (c, h, w (, d)). Defaults to None.
        size (list/tuple of int): size to resize to
            does not include the batch size or number of channels
        order_data (int): interpolation order for data
            (see skimage.transform.resize)
        order_seg (int): interpolation order for seg
            (see skimage.transform.resize)
    """
    target_data = np.ones([data.shape[0]] + list(size))
    if seg is not None:
        target_seg = np.ones([seg.shape[0]] + list(size))
    else:
        target_seg = None

    target_data = resize_multichannel_image(data, size, order_data)
    if seg is not None:
        for c in range(seg.shape[0]):
            target_seg[c] = resize_segmentation(seg[c], size,
                                                order_seg, cval_seg)
    return target_data, target_seg

# from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/cropping.py
def get_bbox_from_mask(mask, outside_value=0):
    """
    Args:
        mask (np.ndarray): shape must be 3D or greater. Assumes that the first
            three dimensions are the spatial dims.
    """
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def expand_bbox_lbub(bbox_dims, length: int = 256):
    """
    Symmetrically expanding a part of a bounding box to `length`.
    Args:
        bbox_dims (list): [lower bound, upper bound]
        length (int): to expand the lower bound and upper bound to
    """
    current_len = bbox_dims[1] - bbox_dims[0]
    if current_len >= length:
        return bbox_dims
    else:
        diff = length-current_len
        if diff % 2 == 0:
            return [bbox_dims[0]-diff//2, bbox_dims[1]+diff//2]
        elif diff % 2 == 1:
            # when odd, expanding the bbox more on the max-side
            # - no particular reason, the offset is just 1 pixel anyways
            return [bbox_dims[0]-diff//2-1, bbox_dims[1]+diff//2]

def expand_bbox(total_bbox, bbox_lengths=[None, 256, 256]):
    """
    Symmetrically expands bounding box coordinates.
    """
    assert len(total_bbox) == 3, \
        "Must be 3D."
    expanded_bbox = []
    for length, lbub in zip(bbox_lengths, total_bbox):
        if length is None:
            expanded_bbox.append(lbub)
        else:
            expanded = expand_bbox_lbub(lbub, length=length)
            expanded_bbox.append(expanded)
    return expanded_bbox

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def resize_bbox(bbox, dims_ratio):
    """
    Resizes the bbox for each axis specified in `dims_ratio`
    Args:
        bbox (list): list of [lower bound, upper bound] sub-lists
        dims_ratio (list): ratios of raw/resized (same length as bbox)
            dims_ratio = np.array(raw_x.shape) / np.array(resized_x.shape)
    """
    assert len(bbox) == len(dims_ratio), \
        "The number of lbub sub-lists must be equal to the number of scale factors in dims_ratio."
    dims_ratio_multiply = np.array(list(zip(dims_ratio, dims_ratio))) # for multiplying to both the lb and ub coords
    scaled_actual_bbox = (actual_bbox * dims_ratio_multiply).astype(np.int32)
    return scaled_actual_bbox
