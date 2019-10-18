
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy

from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

pb_file = "../models/advanced_east.pb"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

with tf.gfile.GFile(pb_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():

    input_bytes = tf.placeholder(tf.string, shape=(), name='input_images')
    decoded_image = tf.image.decode_jpeg(tf.reshape(input_bytes, []), channels=3)
    decoded_image = tf.cast(decoded_image, tf.float32)
    decoded_image = decoded_image / 127.5
    decoded_image = decoded_image - 1.
    decoded_image = tf.expand_dims(decoded_image, 0)
    print(decoded_image)
    input_map = {}
    input_map.setdefault("input_img:0", decoded_image)

    tf.import_graph_def(graph_def, input_map=input_map)
    # tf.import_graph_def(graph_def)

    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    embeddings = graph.get_tensor_by_name("import/east_detect/concat:0")

    with tf.Session() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder("./saved_model_path")

        model_input = tf.saved_model.utils.build_tensor_info(input_bytes)
        # model_phase_train = tf.saved_model.utils.build_tensor_info(phase_train_placeholder)
        model_embeddings = tf.saved_model.utils.build_tensor_info(embeddings)

        model_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'image_bytes': model_input
            },
            outputs={
                'embedding': model_embeddings
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True
        )

        builder.save()
