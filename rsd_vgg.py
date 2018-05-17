# Author: Alexander Decurnou

"""
    This is an implementation of the real-time salient object detection (RSD)
    network proposed by Najibi et. al (2017). This particular version uses the
    convolutional layers of VGG16 in order to calculate the feature maps for
    object classes, and finishes with two branches: saliency map prediction
    and subitizing. The first branch produces a rough saliency map and the
    second branch predicts the number of salient objects.
"""

import tflearn
import numpy as np
import math


def rsd_vgg(incoming, num_classes):

    x = tflearn.conv_2d(incoming, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    sal_pred, subitizing = branch(x, num_classes)

    return sal_pred, subitizing


def branch(incoming, num_classes):
    """
        Returns the two branches specific to the RSD model: a branch that
        outputs a saliency map and a branch that predicts the amount of
        salient objects, given the map.
            incoming:       (Tensor)    incoming tensor from network
            num_classes:    (Integer)   number of labels to predict
    """

    sal_pred = tflearn.conv_2d(incoming, 80, 3, activation='relu', scope='conv_s1')
    sal_pred = tflearn.conv_2d(sal_pred, 80, 1, activation='sigmoid', scope='conv_s2')

    subitizing = tflearn.fully_connected(incoming, 4096, activation='relu', scope='fc6')
    subitizing = tflearn.fully_connected(subitizing, 4096, activation='relu', scope='fc7')
    subitizing = tflearn.fully_connected(subitizing, num_classes, activation='softmax', scope='fc8', restore=False)

    return sal_pred, subitizing
