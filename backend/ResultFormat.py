# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ResultFormat(object):

    def __init__(self):
        self.res = {
            "code": 200,
            "message": "OK"
        }

    def update_key(self, key, value):
        self.res.update({key: value})

    def update_code(self, code):
        self.res.update({"code": code})

    def update_message(self, message):
        self.res.update({"message": message})

    def add_new_key(self, key, value):
        self.res.setdefault(key, value)

    def remove_key(self, key):
        self.res.pop(key)

    def get_res(self):
        return self.res
