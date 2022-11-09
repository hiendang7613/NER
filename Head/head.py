import tensorflow as tf

from Backbone.RNN import BiLSTM

class Head(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Head, self).__init__(**kwargs)
        pass

    def call(self, inputs, training=False) -> tf.Tensor:
        pass


class HeadFactory(object):
    def __init__(self):
        super(HeadFactory, self).__init__()
        pass

    '''
    ## getHead parameters
    config = {
        'head_type' : head_type
        'num_classes' : num_classes
    }'''
    @staticmethod
    def getHead(self, config):
        head = None
        # if head_type == 'head_1':  ### num_classes
        #     head = BiLSTM(include_top=False)
        # elif backbone_type == 'backbone_1':
        #     output = BiLSTM(include_top=False)(input_tensor)
        # else:
        #     print('Head type not match!')
        return head


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