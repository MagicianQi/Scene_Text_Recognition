# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from utils import rotate
from utils.nms import nms
from settings import *


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def resize_image(im, max_img_size=MAX_IMAGE_SIZE):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def get_crop_images(feature_map, im, pixel_threshold=0.9, quiet=True):
    shape = im.size
    d_wight, d_height = resize_image(im, MAX_IMAGE_SIZE)
    scale_ratio_w = d_wight / im.width
    scale_ratio_h = d_height / im.height
    y = feature_map
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = list(map(int, rescaled_geo_list))
            ploy = [[txt_item[0], txt_item[1]], [txt_item[6], txt_item[7]],
                    [txt_item[4], txt_item[5]], [txt_item[2], txt_item[3]]]
            txt_items.append(ploy)
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    crop_images, ploys = rotate.rotate_img(txt_items, np.array(im))
    return crop_images, ploys, shape