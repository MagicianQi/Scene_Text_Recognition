import sys

sys.path.append("../")
from keras import backend as K
from keras.layers import (Conv2D, BatchNormalization, MaxPool2D, Input, Permute, Reshape, Dense, LeakyReLU, Activation,
                          Bidirectional, LSTM, TimeDistributed, Lambda, CuDNNLSTM)
from keras.models import Model
from keras.layers import ZeroPadding2D
from keras.activations import relu
import tensorflow as tf
from utils.keys import alphabetChinese

def keras_crnn(imgH, nc, nclass, nh, leakyRelu=False, lstmFlag=True):
    """
    keras crnn

    """
    data_format = 'channels_first'
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    # InputLayer = Input(shape=(1,), dtype="string", name='imgInput')
    # imgInput = Lambda(lambda x: tf.reshape(x, []))(InputLayer)
    # # imgInput = Lambda(lambda x: tf.decode_base64(x))(imgInput)
    # imgInput = Lambda(lambda x: tf.image.decode_jpeg(x, channels=1))(imgInput)
    # imgInput = Lambda(lambda x: tf.cast(x, tf.float32))(imgInput)
    # imgInput.set_shape([None, imgH, 1])
    # imgInput = Lambda(lambda x: tf.transpose(x, [2, 1, 0]))(imgInput)
    # imgInput = Lambda(lambda x: tf.expand_dims(x, 0))(imgInput)

    imgInput = Input(shape=(1, imgH, None), name='imgInput')
    print(imgInput)

    def convRelu(i, batchNormalization=False, x=None):
        ##padding: one of `"valid"` or `"same"` (case-insensitive).
        ##nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        if leakyRelu:
            activation = LeakyReLU(alpha=0.2)
        else:
            activation = Activation(relu, name='relu{0}'.format(i))

        x = Conv2D(filters=nOut,
                   kernel_size=ks[i],
                   strides=(ss[i], ss[i]),
                   padding='valid' if ps[i] == 0 else 'same',
                   dilation_rate=(1, 1),
                   activation=None, use_bias=True, data_format=data_format,
                   name='cnn.conv{0}'.format(i)
                   )(x)

        if batchNormalization:
            ## torch nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            x = BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1, name='cnn.batchnorm{0}'.format(i))(x)

        x = activation(x)
        return x

    x = imgInput
    x = convRelu(0, batchNormalization=False, x=x)

    # x = ZeroPadding2D(padding=(0, 0), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(0), padding='valid', data_format=data_format)(x)

    x = convRelu(1, batchNormalization=False, x=x)
    # x = ZeroPadding2D(padding=(0, 0), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(1), padding='valid', data_format=data_format)(x)

    x = convRelu(2, batchNormalization=True, x=x)
    x = convRelu(3, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(2),
                  data_format=data_format)(x)

    x = convRelu(4, batchNormalization=True, x=x)
    x = convRelu(5, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(3),
                  data_format=data_format)(x)
    x = convRelu(6, batchNormalization=True, x=x)

    x = Permute((3, 2, 1))(x)

    x = Reshape((-1, 512))(x)

    out = None
    if lstmFlag:
        x = Bidirectional(CuDNNLSTM(nh, return_sequences=True))(x)
        # x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
        #                        recurrent_activation='sigmoid'))(x)
        x = TimeDistributed(Dense(nh))(x)
        x = Bidirectional(CuDNNLSTM(nh, return_sequences=True))(x)
        # x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
        #                        recurrent_activation='sigmoid'))(x)
        out = TimeDistributed(Dense(nclass))(x)
        end = Lambda(lambda x: tf.argmax(x, axis=2))(out)
    else:
        out = Dense(nclass, name='linear')(x)
    # out = Reshape((-1, nclass),name='out')(out)

    return Model(imgInput, end)


def build_model():
    alphabet = alphabetChinese
    nclass = len(alphabet) + 1
    return keras_crnn(32, 1, nclass, 256, leakyRelu=False, lstmFlag=True)


def save_model_to_serving(model, export_path='prod_models'):
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    model_input = tf.saved_model.utils.build_tensor_info(model.input)
    # model_phase_train = tf.saved_model.utils.build_tensor_info(phase_train_placeholder)
    model_embeddings = tf.saved_model.utils.build_tensor_info(model.output)

    model_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'input_image': model_input
        },
        outputs={
            'output': model_embeddings
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        K.get_session(),
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True
    )

    builder.save()


if __name__ == '__main__':
    model = build_model()
    checkpoint_filepath = '../models/ocr-lstm-keras.h5'
    model.load_weights(checkpoint_filepath)
    export_path = "./crnn/1"
    save_model_to_serving(model, export_path)
