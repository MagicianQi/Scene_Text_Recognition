# Scene_Text_Recognition

* Text detection model：Advanced EAST,code is from：https://github.com/huoyijie/AdvancedEAST
* Text recognition model：CRNN，code is from：https://github.com/chineseocr/chineseocr

## Intro

* Advanced EAST and CRNN are deploying with TF-Serving。
* EAST TF-Serving receive image base64，much faster than receiving images directly。
* CRNN model: Change LSTM to CUDNNLSTM, speed increased by about 4 times.

## The environment

* docker - https://docs.docker.com/install/linux/docker-ce/ubuntu/
* nvidia-docker - https://github.com/NVIDIA/nvidia-docker

## Run

* `sudo docker pull tensorflow/serving:1.13.0-gpu`
* `wget https://github.com/MagicianQi/Scene_Text_Recognition/releases/download/v1.0/flask-uwsgi-python-centos.tar && sudo docker load -i flask-uwsgi-python-centos.tar`
* `git clone https://github.com/MagicianQi/Scene_Text_Recognition`
* `cd Scene_Text_Recognition && wget https://github.com/MagicianQi/Scene_Text_Recognition/releases/download/v1.0/models.zip && unzip models.zip`
* Specify the GPU ID: `vim deploy.sh`
    1.https://github.com/MagicianQi/Scene_Text_Recognition/blob/master/deploy.sh#L10
    2.https://github.com/MagicianQi/Scene_Text_Recognition/blob/master/deploy.sh#L27
* Specify the model absolute path：`vim deploy.sh`
    1.https://github.com/MagicianQi/Scene_Text_Recognition/blob/master/deploy.sh#L12
    2.https://github.com/MagicianQi/Scene_Text_Recognition/blob/master/deploy.sh#L29
* `bash deploy.sh`
* Test: `curl localhost:8080`

## Test

* Address of the test：https://IP:8080/test
* Other API：https://github.com/MagicianQi/Scene_Text_Recognition/blob/master/flask_server.py
