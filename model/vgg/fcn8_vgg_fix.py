from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_fcn import fcn8_vgg

import tensorflow as tf


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    vgg_fcn = fcn8_vgg.FCN8VGG()

    vgg_fcn.build(images, train=train, num_classes=2, random_init_fc8=True)

    return vgg_fcn.upscore32