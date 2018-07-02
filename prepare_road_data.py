import os
import sys
import logging
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave

LABEL_THRESHOLD = 150
TRAINING_DATA_URL = 'https://drive.google.com/uc?export=download&id=1XU0YQkH5jEmg7OBXsH6uX1shCd7a2gRD'
TEST_DATA_URL = 'https://drive.google.com/uc?export=download&id=195--p90lFpiqcdtNGpUq2RsGsKM-dZ5j'
VGG_URL = 'ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy'
VGG_DIR = 'weights/'
DATA_DIR = 'DATA/'
TRAIN_DIR = 'training/'
TEST_DIR = 'testing/'
IMAGE_DIR = 'images/'
GT_DIR = 'groundtruth/'
PROC_GT_DIR = 'processed_groundtruth/'
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'
TEST_FILE = 'test.txt'
SPLIT = 80


# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def maybe_download_and_extract(data_dir=DATA_DIR,
                               train_dir=TRAIN_DIR,
                               train_data_url=TRAINING_DATA_URL,
                               test_dir=TEST_DIR,
                               test_data_url=TEST_DATA_URL,
                               vgg_url=VGG_URL,
                               vgg_dir=VGG_DIR):
    """ Downloads, extracts and prepairs data.

    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    sys.path.insert(1, 'incl')
    import tensorvision.utils as utils
    import zipfile
    import tarfile

    train_data_zip = os.path.join(data_dir, 'training.zip')
    train_data_dir = os.path.join(data_dir, train_dir)
    if not os.path.exists(train_data_dir):
        # Download Train Road DATA
        if train_data_url == '':
            logging.error("Data URL for Training Road Data not provided.")
            exit(1)

        logging.info("Downloading Training Road Data.")
        filepath = utils.download(train_data_url, data_dir)
        os.rename(filepath, train_data_zip)
        # Extract and prepare Satellite DATA
        logging.info("Extracting training data.")
        zipfile.ZipFile(train_data_zip, 'r').extractall(data_dir)

    test_data_tar = os.path.join(data_dir, 'test_images.tar.gz')
    test_data_dir = os.path.join(data_dir, test_dir)
    if not os.path.exists(test_data_dir):
        # Download Test Road DATA
        if test_data_url == '':
            logging.error("Data URL for Testing Road Data not provided.")
            exit(1)

        logging.info("Downloading Testing Road Data.")
        filepath = utils.download(test_data_url, data_dir)
        os.rename(filepath, test_data_tar)
        # Extract and prepare Satellite DATA
        logging.info("Extracting testing data.")
        os.mkdir(test_data_dir)
        tar = tarfile.open(test_data_tar, 'r:gz')
        tar.extractall(path=test_data_dir)
        tar.close()
        os.rename(os.path.join(test_data_dir, 'test_images'), os.path.join(test_data_dir, 'images'))

    vgg_weights = os.path.join(data_dir, vgg_dir, 'vgg16.npy')
    if not os.path.exists(vgg_weights):
        # Download VGG DATA
        logging.info("Downloading VGG weights.")
        vgg_path = os.path.join(data_dir, vgg_dir)
        os.makedirs(vgg_path)
        utils.download(vgg_url, vgg_path)
    return


def prepare_test_data(data_dir=os.path.join(DATA_DIR, TEST_DIR),
                      image_dir=IMAGE_DIR,
                      gt_dir=GT_DIR,
                      test_file=TEST_FILE):

    logging.info("Preparing testing data.")
    full_image_path = os.path.join(data_dir, image_dir)
    image_files = sorted([f for f in os.listdir(full_image_path) if os.path.isfile(os.path.join(full_image_path, f))])
    full_gt_path = os.path.join(data_dir, gt_dir)
    if not os.path.exists(full_gt_path):
        os.makedirs(full_gt_path)

    with open(os.path.join(data_dir, test_file), 'w') as f:
        for i in image_files:
            im = imread(os.path.join(full_image_path, i), mode='RGB')
            im[im > LABEL_THRESHOLD] = 255
            im[im <= LABEL_THRESHOLD] = 0
            imsave(os.path.join(full_gt_path, i), im)
            f.write('{}{} {}{}\n'.format(image_dir, i, gt_dir, i))


def prepare_train_data(data_dir=os.path.join(DATA_DIR, TRAIN_DIR),
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

    # Process and prepare training data
    prepare_train_data()

    # Process and prepare testing data
    prepare_test_data()


if __name__ == '__main__':
    main()
