"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

"""
import math
import os
import numpy as np
import pandas as pd
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import sys
from keras.preprocessing import image as KImage

ROOT_DIR = os.path.abspath("Mask_RCNN/")

sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

COCO_MODEL_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################


class ImatirealistConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "iMaterialist"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 6

    GPU_COUNT = 1

    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    # Number of classes (including background)
    NUM_CLASSES = 1 + 46  # iMaterialist has 46 classes

    BACKBONE = "resnet50"


############################################################
#  Dataset
############################################################

class ImaterialistDataset(utils.Dataset):

    def __init__(self, data_frame, base_dir, image_ids_key, image_dir):
        super(ImaterialistDataset, self).__init__()
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.data_frame = data_frame
        self.load_imaterialist(image_ids_key)
        self.prepare()

    def load_imaterialist(self, image_ids_key):
        """Load Image Ids and classIds
        """
        label_description_path = os.path.join(self.base_dir, 'label_descriptions.json')
        with open(label_description_path) as f:
            label_description = json.load(f)
        for label in label_description['categories']:
            self.add_class('imaterialist', label['id'] + 1, label['name'])

        train_image_ids_path = os.path.join(self.base_dir, 'imageIds.json')
        with open(train_image_ids_path) as tif:
            train_image_ids = json.load(tif)
        for img_id in train_image_ids[image_ids_key]:
            self.add_image(
                "imaterialist",
                img_id,
                os.path.join(self.image_dir, img_id + '.jpg'),
                width=self.data_frame.loc[self.data_frame.ImageId == img_id, 'Width'].iloc[0],
                height=self.data_frame.loc[self.data_frame.ImageId == img_id, 'Height'].iloc[0]
            )

    def load_mask(self, image_id):
        """Load Mask

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        data = self.data_frame[self.data_frame.ImageId == info['id']]
        encodedPixels = data.EncodedPixels.iloc[0]
        class_ids = np.asarray(data.ClassId.iloc[0], dtype='int32')
        masks = np.zeros((info['height'], info['width'], len(class_ids)))
        for idx, ep in enumerate(encodedPixels):
            ep = np.array(ep.split(' '), dtype='int32')
            ep = ep.reshape(-1, 2)
            for pix in ep:
                y = (int(pix[0] % info['height'])) - 1
                x = math.floor(pix[0] / info['height'])
                y1 = y + pix[1]
                masks[y:y1, x, idx] = 1
        return masks, class_ids

    def get_class_id(self, image_id):
        info = self.image_info[image_id]
        data = self.data_frame[self.data_frame.ImageId == info['id']]
        return data.ClassId.iloc[0]

    def load_image(self, image_id):
        path = self.image_info[image_id]['path']
        img = KImage.load_img(path)
        return np.array(img)


############################################################
#  Training
############################################################


def train(base_dir, img_dir):
    log_dir = os.path.join(base_dir, 'logs')

    config = ImatirealistConfig()
    config.display()

    data_frame_path = os.path.join(base_dir, 'train.csv')
    train_data = pd.read_csv(data_frame_path)

    # change classId dtype
    train_data.ClassId = train_data.ClassId.astype('category')
    train_data.ClassId = train_data.ClassId.cat.codes

    # Drop Attribute Ids
    train_data = train_data.drop('AttributesIds', axis=1)

    # group data by image id
    train_df = train_data.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
    size_df = train_data.groupby('ImageId')['Height', 'Width'].mean()
    train_df = train_df.join(size_df, on='ImageId')
    train_df = train_df.reset_index()
    train_data = None

    train_dataset = ImaterialistDataset(train_df, base_dir, 'train_image', img_dir)
    train_dataset.prepare()

    val_dataset = ImaterialistDataset(train_df, base_dir, 'val_image', img_dir)
    val_dataset.prepare()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=log_dir)

    weight_path = model.get_imagenet_weights()
    model.load_weights(weight_path, by_name=True)

    augmentation = imgaug.augmenters.Sometimes(0.5, [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
    ])
    class_ids_to_augment = [27, 3, 44, 11, 30, 5, 40, 38, 45, 41, 12, 26, 20]
    # Training - Stage 1
    print("Training network heads")
    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation,
                class_id_to_augment=class_ids_to_augment)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(train_dataset, val_dataset,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation,
                class_id_to_augment=class_ids_to_augment)


def getData(base_dir):
    data_frame_path = os.path.join(base_dir, 'train.csv')
    train_data = pd.read_csv(data_frame_path)

    # change classId dtype
    train_data.ClassId = train_data.ClassId.astype('category')
    train_data.ClassId = train_data.ClassId.cat.codes

    # Drop Attribute Ids
    train_data = train_data.drop('AttributesIds', axis=1)

    # group data by image id
    train_df = train_data.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
    size_df = train_data.groupby('ImageId')['Height', 'Width'].mean()
    train_df = train_df.join(size_df, on='ImageId')
    train_df = train_df.reset_index()
    train_data = None

    img_dir = os.path.join(base_dir, 'train')

    train_dataset = ImaterialistDataset(train_df, base_dir, 'train_image', img_dir)
    return train_dataset
