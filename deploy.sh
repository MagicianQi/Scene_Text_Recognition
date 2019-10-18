#!/usr/bin/env bash

# Text Recognition Model

sudo docker run \
--runtime=nvidia \
--name text_recognition \
--restart always \
-d \
-e CUDA_VISIBLE_DEVICES=2 \
-p 8500:8500 -p 8501:8501 \
--mount type=bind,source=/home/qishuo/PycharmProjects/spam_OCR_TF/models/crnn,target=/models/crnn \
-t --entrypoint=tensorflow_model_server tensorflow/serving:1.13.0-gpu \
--port=8500 --rest_api_port=8501 \
--model_name=crnn \
--model_base_path=/models/crnn \
--enable_batching \
--per_process_gpu_memory_fraction=0.4

# Text Detection Model

sudo docker run \
--runtime=nvidia \
--name text_detection \
--restart always \
-d \
-e CUDA_VISIBLE_DEVICES=2 \
-p 8502:8502 -p 8503:8503 \
--mount type=bind,source=/home/qishuo/PycharmProjects/spam_OCR_TF/models/east,target=/models/east \
-t --entrypoint=tensorflow_model_server tensorflow/serving:1.13.0-gpu \
--port=8502 --rest_api_port=8503 \
--model_name=east \
--model_base_path=/models/east \
--enable_batching \
--per_process_gpu_memory_fraction=0.4

# Build flask docker image

sudo docker build -t spam_ocr .

# Sensitive Model Flask

sudo docker run \
--link text_detection:text_detection \
--link text_recognition:text_recognition \
-p 8080:8080 -itd spam_ocr

