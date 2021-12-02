import os
import random
import numpy as np


DATA_PATH = "data/NCT-CRC-HE-100K"
SAMPLE_SIZE_PER_CLASS = 100

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir, DATA_PATH)

print("DATA PATH", DATA_PATH)

for split in ["train", "test"]:
    random_imgs = []
    random_labels = []
    for cls in os.listdir(DATA_PATH): # for each class
        class_path = os.path.join(DATA_PATH, cls)
        if os.path.isdir(class_path): # only loop through folders
            class_imgs = os.listdir(class_path)
            # sample k images from each class
            random_class_imgs = random.sample(class_imgs, SAMPLE_SIZE_PER_CLASS)
            # prepend the data path to each img
            random_class_imgs = [os.path.join(class_path, img) for img in random_class_imgs]
            class_labels = [cls] * SAMPLE_SIZE_PER_CLASS
            random_imgs += random_class_imgs
            random_labels += class_labels
            # TODO we might want to shuffle the images, but remember to align images with labels

    filelist_imgs_path = os.path.join(DATA_PATH, f"{split}_images.npy")
    filelist_labels_path = os.path.join(DATA_PATH, f"{split}_labels.npy")
    np.save(filelist_imgs_path, np.array(random_imgs))
    np.save(filelist_labels_path, np.array(random_labels))


