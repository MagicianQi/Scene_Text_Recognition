# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import base64
import random
import requests
import numpy as np

from PIL import Image, ImageSequence

from backend.RedisConnection import RedisConnection
from utils.east_util import get_crop_images, resize_image
from utils.keys import alphabetChinese
from utils.crnn_util import resizeNormalize, strLabelConverter
from utils.other_util import Logger

from settings import *

logger = Logger(ERROR_LOG_FILE_PATH, "error")


class OCRModel(object):

    def __init__(self,
                 east_api_url=EAST_API_URL,
                 crnn_api_url=CRNN_API_URL,
                 max_image_size=MAX_IMAGE_SIZE):
        self.east_api_url = east_api_url
        self.crnn_api_url = crnn_api_url
        self.alphabet = alphabetChinese
        self.max_image_size = max_image_size
        self.redis_connection = RedisConnection()

    def text_detection_feature_map(self, image):
        im = image.convert('RGB')
        d_wight, d_height = resize_image(im, self.max_image_size)
        img = im.resize((d_wight, d_height), Image.NEAREST)
        img = np.array(img)
        image = Image.fromarray(img, mode="RGB")
        image_pil = io.BytesIO()
        image.save(image_pil, format='JPEG')
        img_b64 = base64.b64encode(image_pil.getvalue()).decode('utf-8')
        post_json = {
            "instances": [{
                "b64": img_b64
            }]
        }
        response = requests.post(self.east_api_url, data=json.dumps(post_json))
        response.raise_for_status()
        prediction = response.json()["predictions"]
        return prediction

    def text_recognition_vector(self, crop_image):
        image = Image.fromarray(crop_image).convert('L')
        image = resizeNormalize(image, 32)
        image = image.astype(np.float32)
        image = np.array([image])
        post_json = {
            "instances": [{
                "input_image": image.tolist()
            }]
        }
        response = requests.post(self.crnn_api_url, data=json.dumps(post_json))
        response.raise_for_status()
        prediction = response.json()["predictions"]
        raw = strLabelConverter(prediction[0], self.alphabet)
        return raw

    def get_img_from_redis(self, key):
        value = self.redis_connection.get_image_by_key(key)
        data = base64.b64decode(value)
        image = io.BytesIO(data)
        image = Image.open(image)
        return image

    @staticmethod
    def get_img_from_url(image_url):
        response = requests.get(image_url)
        response = response.content
        bytes_obj = io.BytesIO(response)
        image = Image.open(bytes_obj)
        return image

    @staticmethod
    def process_gif(image_gif):
        frame_list = [frame.copy() for frame in ImageSequence.Iterator(image_gif)]
        frame_list = [frame_list[i] for i in range(len(frame_list)) if i % GIF_FRAME_INTERVAL == 0]
        if len(frame_list) > GIF_MAX_FRAME:
            return random.sample(frame_list, GIF_MAX_FRAME)
        else:
            return frame_list

    def get_result(self, image):
        try:
            feature_map = self.text_detection_feature_map(image)
            crop_images, ploys, im_shape = get_crop_images(np.array(feature_map), image)
            result_list = []
            for crop_image in crop_images:
                result_list.append(self.text_recognition_vector(crop_image))
            return result_list, ploys
        except Exception as e:
            logger.out_print([e])
            return [], []

    def get_result_from_redis(self, image_id):
        return self.get_result(self.get_img_from_redis(image_id))

    def get_result_from_url(self, image_url):
        image = self.get_img_from_url(image_url)
        return self.get_result(image)


if __name__ == "__main__":
    ocr_model = OCRModel()
    print(ocr_model.get_result_from_url("http://172.31.1.51:8000/004.jpg"))
