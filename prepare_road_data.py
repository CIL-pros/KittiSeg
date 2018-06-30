import os
import sys
import logging
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave

LABEL_THRESHOLD = 150
TRAINING_DATA_URL = 'https://drive.google.com/uc?export=download&id=1XU0YQkH5jEmg7OBXsH6uX1shCd7a2gRD'
VGG_URL = 'ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy'
VGG_DIR = 'weights/'
DATA_DIR = 'DATA/'
TRAIN_DIR = 'training/'
IMAGE_DIR = 'images/'
GT_DIR = 'groundtruth/'
PROC_GT_DIR = 'processed_groundtruth/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
SPLIT = 80


# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)


def maybe_download_and_extract(data_dir=DATA_DIR,
                               train_dir=TRAIN_DIR,
                               data_url=TRAINING_DATA_URL,
                               vgg_url=VGG_URL,
                               vgg_dir=VGG_DIR):
    """ Downloads, extracts and prepairs data.

    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_road_zip = os.path.join(data_dir, 'training.zip')
    vgg_weights = os.path.join(data_dir, vgg_dir, 'vgg16.npy')
    road_data_dir = os.path.join(data_dir, train_dir)

    if os.path.exists(vgg_weights) and os.path.exists(road_data_dir):
        return

    sys.path.insert(1, 'incl')
    import tensorvision.utils as utils
    import zipfile

    # Download Road DATA
    if data_url == '':
        logging.error("Data URL for Road Data not provided.")
        exit(1)

    logging.info("Downloading Road Data.")
    filepath = utils.download(data_url, data_dir)
    os.rename(filepath, data_road_zip)
    # Extract and prepare Satellite DATA
    logging.info("Extracting training data.")
    zipfile.ZipFile(data_road_zip, 'r').extractall(data_dir)

    # Download VGG DATA
    logging.info("Downloading VGG weights.")
    vgg_path = os.path.join(data_dir, vgg_dir)
    os.makedirs(vgg_path)
    utils.download(vgg_url, vgg_path)
    return


def prepare_data(data_dir=os.path.join(DATA_DIR, TRAIN_DIR),
                 image_dir=IMAGE_DIR,
                 gt_dir=GT_DIR,
                 proc_gt_dir=PROC_GT_DIR,
                 train_file=TRAIN_FILE,
                 val_file=VAL_FILE,
                 split=SPLIT):

    logging.info("Preparing training data.")
    full_image_path = os.path.join(data_dir, image_dir)
    image_files = sorted([f for f in os.listdir(full_image_path) if os.path.isfile(os.path.join(full_image_path, f))])
    full_gt_path = os.path.join(data_dir, gt_dir)
    gt_files = sorted([f for f in os.listdir(full_gt_path) if os.path.isfile(os.path.join(full_gt_path, f))])

    n_images = len(image_files)
    perm_indices = np.random.permutation(n_images).astype(np.int32)
    split_index = int(split / 100.0 * n_images)

    train_images = [image_files[i] for i in perm_indices[:split_index]]
    train_gts = [gt_files[i] for i in perm_indices[:split_index]]

    val_images = [image_files[i] for i in perm_indices[split_index:]]
    val_gts = [gt_files[i] for i in perm_indices[split_index:]]

    full_proc_gt_dir = os.path.join(data_dir, proc_gt_dir)
    if not os.path.exists(full_proc_gt_dir):
        os.makedirs(full_proc_gt_dir)

    with open(os.path.join(data_dir, train_file), 'w') as f:
        for i, g in zip(train_images, train_gts):
            im = imread(os.path.join(full_gt_path, g), mode='RGB')
            im[im > LABEL_THRESHOLD] = 255
            im[im <= LABEL_THRESHOLD] = 0
            imsave(os.path.join(full_proc_gt_dir, g), im)
            f.write('{}{} {}{}\n'.format(image_dir, i, proc_gt_dir, g))

    with open(os.path.join(data_dir, val_file), 'w') as f:
        for i, g in zip(val_images, val_gts):
            im = imread(os.path.join(full_gt_path, g), mode='RGB')
            im[im > LABEL_THRESHOLD] = 255
            im[im <= LABEL_THRESHOLD] = 0
            imsave(os.path.join(full_proc_gt_dir, g), im)
            f.write('{}{} {}{}\n'.format(image_dir, i, proc_gt_dir, g))
    return


def main():
    # Download road data and VGG16 weights
    maybe_download_and_extract()

    # Process and prepare data
    prepare_data()


if __name__ == '__main__':
    main()
