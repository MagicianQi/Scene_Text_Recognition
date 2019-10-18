# 场景文本识别（Scene_Text_Recognition）

* 文本检测模型为Advanced EAST,代码及模型来自：https://github.com/huoyijie/AdvancedEAST
* 文本识别模型为CRNN实现，代码及模型来自：https://github.com/chineseocr/chineseocr

## 简单介绍

* Advanced EAST 以及 CRNN 都使用TF Serving部署。
* EAST TF Serving接收图片base64，比直接接收图像速度快很多。
* CRNN模型中将LSTM改为CUDNNLSTM，速度提升约4倍。
* 使用docker提供服务。

## 环境

* docker - https://docs.docker.com/install/linux/docker-ce/ubuntu/
* nvidia-docker - https://github.com/NVIDIA/nvidia-docker

## How to use

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

## 测试

* 测试：https://IP:8080/test
* 其他接口见：https://github.com/MagicianQi/Scene_Text_Recognition/blob/master/flask_server.py
