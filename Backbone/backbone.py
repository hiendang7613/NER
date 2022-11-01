import tensorflow as tf

from Backbone.LSTM import BiLSTM

class Backbone(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, inputs, training=False) -> tf.Tensor:
        pass


class BackboneFactory(tf.keras.layers.Layer):
    def __init__(self):
        pass

    '''
    ## getBackbone parameters
    config = {
        'backbone_type' : '..'
        'input_shape' : (width, height, chanels)
    } '''
    @staticmethod
    def getBackbone(self, config):
        input_tensor = tf.keras.layers.Input(shape=config['input_shape'])
        if config['backbone_type'] == 'backbone_1':
            output = BiLSTM(include_top=False)(input_tensor)
        # elif backbone_type == 'backbone_1':
        #     output = BiLSTM(include_top=False)(input_tensor)
        backbone = tf.keras.Model(input_tensor, output)
        return backbone


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