# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from redis import Redis

from settings import *


class RedisConnection(object):

    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD):
        self.redis = Redis(
            host=host,
            port=port,
            password=password)

    def get_image_by_key(self, key):
        return self.redis.get(key)
