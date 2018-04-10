# IDRiD dataset for Mask R-CNN

import math
import random
import numpy as np
import glob
import scipy.io as sio
import skimage.transform as imtransform
import scipy.ndimage as ndi
import os
from tqdm import tqdm

from config import Config
import utils

image_size = (384, 384,)
mask_area_threshold = 4 # Lesion masks with area <= this threshold are dropped, to save space.

class IdridConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "idrid"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 4 lesions + OD
                         # (MA/Microaneurysm, HE/Hemorrhage, EX/Hard Exudate, SE/Soft Exudate)

    # Use resnet50 instead of resnet101 for backbone, to reduce memory usage
    # BACKBONE = "resnet50"
    # BACKBONE_STRIDES = [4, 8, 16, 32]

    # Images are constant height and width
    IMAGE_MIN_DIM = image_size[0]
    IMAGE_MAX_DIM = image_size[1]

    # # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 256)  # anchor side in pixels

    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 400

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class IdridDataset(utils.Dataset):
    """
    Loads IDRiD dataset from .mat files.
    To pre-load images and masks into memory, pass preload=True when calling __init__().
    This is to prevent re-reading mat file but memory usage might be huge.
    """

    dataset_path = './images/raw/DR/'
    lesion_types = [
        'MA',
        'HE',
        'EX',
        'SE',
        'OD',
    ]

    def __init__(self, preload = True, *args, **kwargs):
        self.preload = preload
        if preload:
            # Preloaded image and mask arrays
            self.images = []
            self.masks = []
        else:
            self.image_paths = []
        super().__init__(*args, **kwargs)

    # Returns tuple of (image_data, (lesion_masks, instance_ids))
    def read_mat(self, mat_file_path):
        data = sio.loadmat(mat_file_path)
        image = data['I_cropped']
        ground_truths = data['GT']

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
                lesion_mask = imtransform.resize(lesion_mask, image_size)
                lesion_mask[lesion_mask > 0] = 1
                # Separate lesions into instances
                structure = [[1,1,1],[1,1,1],[1,1,1]] # Consider diagonal pixels as same lesion
                labeled_lesions, num_instances = ndi.label(lesion_mask, structure=structure)
                # Add each instance to class_ids and lesion_masks
                num_below_threshold = 0 # Number of lesions with mask area <= threshold
                for i in range(1, num_instances+1):
                    single_lesion = np.where(labeled_lesions == i, labeled_lesions, 0) # Select only that lesion, zero out others
                    single_lesion[single_lesion > 0] = 1 # Reset elements back to 1 (was i)
                    #Select only lesions with area > mask_area_threshold
                    if np.sum(single_lesion) <= mask_area_threshold:
                        num_below_threshold += 1
                        continue
                    single_lesion = single_lesion.reshape(image_size + (1,))
                    lesion_masks = np.append(lesion_masks, single_lesion, axis=2) # Append to lesion_masks
                    class_ids.append(class_num + 1) #+1 because start from 0

                # print("Found {} {} instances for {}".format(num_instances - num_below_threshold, lesion_type, mat_file_path))

        return (image, (lesion_masks, np.asarray(class_ids,)),)

    def load_idrid(self, subset='train'):
        # Add classes
        for idx, lesion_type in enumerate(self.lesion_types):
            self.add_class("idrid", idx + 1, lesion_type)

        # Add images
        image_list = glob.glob(self.dataset_path + subset + '/*.mat')
        random.shuffle(image_list)
        for image_id, mat_file_path in enumerate(tqdm(image_list)):
            self.add_image("idrid", image_id, mat_file_path)

            if self.preload:
                image_data, mask_data = self.read_mat(mat_file_path)
                self.images.append(image_data)
                self.masks.append(mask_data)

    def load_image(self, image_id):
        if self.preload:
            return self.images[image_id]
        else:
            image_path = self.source_image_link(image_id)
            return self.read_mat(image_path)[0]

    def load_mask(self, image_id):
        if self.preload:
            return self.masks[image_id]
        else:
            image_path = self.source_image_link(image_id)
            return self.read_mat(image_path)[1]
