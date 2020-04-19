import numpy as np
import torch
import cv2

def cam(x, image_size=256):
    cam_img = np.interp(x, (x.min(), x.max()), (0, 255)).astype(np.uint8)
    cam_img = cv2.resize(cam_img, (image_size, image_size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img


def tensor2im(input_image, imtype=np.uint8, use_cam=False, image_size=256):
    """
    Converts a Tensor array into a numpy image array.
    :param input_image: (tensor) the input image tensor array
    :param imtype: (type) the desired type of the converted numpy array
    :return:
    """

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        if use_cam:
            image_numpy = cam(np.transpose(image_numpy, (1, 2, 0)), image_size)
        else:
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)