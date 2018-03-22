# IDRiD dataset for Mask R-CNN

import math
import random
import numpy as np
import glob
import scipy.io as sio
import skimage.transform as imtransform
import os
from tqdm import tqdm

from config import Config
import utils

image_size = (512, 512,)

class IdridConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "idrid"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 4 lesions + OD
                         # (MA/Microaneurysm, HE/Hemorrhage, EX/Hard Exudate, SE/Soft Exudate)

    # Images are constant height and width
    IMAGE_MIN_DIM = image_size[0]
    IMAGE_MAX_DIM = image_size[1]

    # # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 256)  # anchor side in pixels

    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 128

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class IdridDataset(utils.Dataset):
    """
    Loads IDRiD dataset from .mat files.
    Note that images and masks are pre-loaded into arrays to prevent re-reading of mat file.
    """

    dataset_path = './images/raw/DR/'
    lesion_types = [
        'MA',
        'HE',
        'EX',
        'SE',
        'OD',
    ]

    def __init__(self, *args, **kwargs):
        # Preloaded image and mask arrays
        self.images = []
        self.masks = []
        super().__init__(*args, **kwargs)

    def load_idrid(self, subset='train'):
        """Generate the requested number of synthetic images.
        subset: train or test, or a valid folder name
        """
        # Add classes
        for idx, lesion_type in enumerate(self.lesion_types):
            self.add_class("idrid", idx + 1, lesion_type)

        # Add images
        for image_id, mat_file in enumerate(tqdm(glob.glob(self.dataset_path + subset + '/*.mat'))):
            # Read
            data = sio.loadmat(mat_file)
            image = data['I_cropped']
            ground_truths = data['GT']

            self.add_image("idrid", image_id, mat_file) #Note that path is .mat file

            # Resize
            image = imtransform.resize(image, image_size)
            image *= 255

            lesion_masks = [[[] for __ in range(image_size[1])] for _ in range(image_size[0])] #Empty 3d list
            class_ids = []

            for class_num, lesion_type in enumerate(self.lesion_types):
                gt_label = lesion_type + "_mask"
                if ground_truths[gt_label][0][0].shape[0]:
                    # Mask exists
                    lesion_mask = ground_truths[gt_label][0][0]
                    lesion_mask = lesion_mask.reshape(lesion_mask.shape + (1,))
                    lesion_mask = imtransform.resize(lesion_mask, image_size)
                    lesion_mask[lesion_mask > 0] = 1
                    # Add to class_id list 
                    class_ids.append(class_num + 1) #+1 because start from 0
                    lesion_masks = np.append(lesion_masks, lesion_mask, axis=2)

            # Add to self.images and self.masks
            self.images.append(image)
            self.masks.append((lesion_masks, np.asarray(class_ids),))


    # Load image and mask from memory so it's faster
    def load_image(self, image_id):
        return self.images[image_id]

    def load_mask(self, image_id):
        return self.masks[image_id]
