# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import io
import random
import requests

from settings import *
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(img_url, text_list, ploys):
    response = requests.get(img_url)
    response = response.content
    bytes_obj = io.BytesIO(response)
    image = Image.open(bytes_obj)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    count = 0
    font = ImageFont.truetype(FONT_FILE_PATH, FONT_SIZE)
    for ploy in ploys:
        count += 1
        x1, x2, x3, x4 = ploy
        draw.text(x4, text_list[count - 1], font=font, fill=FONT_COLOR)
        draw.line(tuple(x1+x2), fill=LINE_COLOR)
        draw.line(tuple(x2+x3), fill=LINE_COLOR)
        draw.line(tuple(x3+x4), fill=LINE_COLOR)
        draw.line(tuple(x4+x1), fill=LINE_COLOR)
    path = SAVE_TEMP_IMAGE_PATH + random_str(TEMP_IMAGE_RANDOM_NAME_LENGTH) + ".png"
    image.save(path)
    return "/" + path


def random_str(length):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    salt = ''
    for i in range(length):
        salt += random.choice(chars)
    return salt


class Logger(object):

    def __init__(self, file_path, name="logger"):
        """
        Initialize the log class
        :param file_path: Log file path
        """
        self.logger = logging.getLogger(name)
        handler = logging.FileHandler(filename=file_path)
        self.logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def out_print(self, in_list, separator=","):
        """
        Print output to file
        :param in_list: Input data list
        :param separator: Separator for each item
        :return: None
        """
        self.logger.info("[{}]".format(separator.join(list(map(str, in_list)))))


if __name__ == "__main__":
    pass
