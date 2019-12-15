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
