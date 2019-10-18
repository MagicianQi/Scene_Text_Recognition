# -*- coding: utf-8 -*-

import flask
import json
import time

from flask import render_template, request

from backend.OCRModel import OCRModel
from backend.ResultFormat import ResultFormat
from utils.other_util import draw_boxes
from utils.other_util import Logger

from settings import *

app = flask.Flask(__name__)
ocr_model = OCRModel()
logger = Logger(RECORD_LOG_FILE_PATH, "record")


@app.route("/")
def homepage():
    return "Welcome to the SOUL Face Verification REST API!"


@app.route("/health")
def health_check():
    return "OK"


@app.route("/api/predict/text", methods=["POST", "GET"])
def predict_sensitive():
    res = ResultFormat()
    if flask.request.method == "POST":
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        if "imgId" in data:
            res.add_new_key("imgID", data["imgId"])
            start = time.time()
            text, _ = ocr_model.get_result_from_redis(data["imgId"])
            res.add_new_key("predictions", text)
            res.add_new_key("time", time.time() - start)
        elif "imgUrl" in data:
            res.add_new_key("imgUrl", data["imgUrl"])
            start = time.time()
            text, _ = ocr_model.get_result_from_url(data["imgUrl"])
            res.add_new_key("predictions", text)
            res.add_new_key("time", time.time() - start)
        else:
            res.update_code(400)
            res.update_message("Bad Request.")
    logger.out_print([res.get_res()])
    return flask.jsonify(res.get_res())


@app.route("/test", methods=["GET"])
def test():
    return render_template("home.html")


@app.route("/api/draw", methods=["GET"])
def draw():
    if flask.request.method == "GET":
        img_url = request.args.get('ImageUrl')
        text, ploys = ocr_model.get_result_from_url(img_url)
        return draw_boxes(img_url, text, ploys)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8585)
