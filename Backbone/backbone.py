import tensorflow as tf
from Config import train_config

from Backbone.RNN import BiLSTM

class Backbone(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Backbone, self).__init__(**kwargs)
        pass

    def call(self, inputs, training=False) -> tf.Tensor:
        pass


class BackboneFactory(object):
    def __init__(self):
        super(BackboneFactory, self).__init__()
        pass

    '''
    ## getBackbone parameters
        'backbone_type' 
        'input_shape' : (width, height, chanels)
        'embedding_size'
    '''
    @staticmethod
    def getBackbone(self, backbone_type, embedding, input_shape=train_config.input_shape, embedding_size=train_config.embedding_size):
        input_tensor = tf.keras.layers.Input(shape=input_shape)
        x = embedding(input_tensor)
        if backbone_type == 'BiLSTM':
            output = BiLSTM(embedding_size)(x)
        # elif backbone_type == 'backbone_1':
        #     output = BiLSTM(include_top=False)(input_tensor)
        backbone = tf.keras.Model(input_tensor, output)
        # output = tf.keras.layers.Dense(embedding_size)(backbone.output)
        return backbone.output


# if __name__ == '__main__':
#     """
#         - ResNet_v1_101
#         - ResNet_v1_34
#         - Resnet_tf
#         - Vgg16
#     """
#     input_shape = 250
#     model = MyModel(type_backbone='Resnet_tf',
#                     input_shape=input_shape,
#                     header=ArcHead(num_classes=1000, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
#     model.build(input_shape=(None, input_shape, input_shape, 3))
#     print(model.summary())
#
#     x = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))
#     out = model(x, training=True)
#
#     print(f"input: {x}")
#     print(f"output: {out}")
#     print("DONE ...")