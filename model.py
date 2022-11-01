import tensorflow as tf
from Backbone.backbone import BackboneFactory
from Head.head import HeadFactory

class MyModel(tf.keras.Model):
    def __init__(self, backbone, head):
        super(MyModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=False):
        out = self.backbone(inputs, training=training)
        out = self.head(out, training=training)
        return out

class ModelFactory(object):
    def __init__(self):
        pass

    '''
    ## getModel parameters
    modelConfig = {
        'backbone_type': backbone_type,
        'input_shape' : (width, height, chanels)
        'head_type' : head_type,
        'num_classes' : num_classes
    }'''
    @staticmethod
    def getModel(self, modelConfig backbone_type, head_type, input_shape):
        backboneConfig = {
            'backbone_type':
            'input_shape':
        }
        backbone = BackboneFactory.getBackbone(config={'modelConfig'})
        head = HeadFactory.getHead(head_type)
        model = MyModel(backbone, head)
        return model


if __name__ == '__main__':
    """
        - ResNet_v1_101
        - ResNet_v1_34
        - Resnet_tf
        - Vgg16
    """
    input_shape = 250
    model = MyModel(type_backbone='Resnet_tf',
                    input_shape=input_shape,
                    header=ArcHead(num_classes=1000, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    model.build(input_shape=(None, input_shape, input_shape, 3))
    print(model.summary())

    x = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))
    out = model(x, training=True)

    print(f"input: {x}")
    print(f"output: {out}")
    print("DONE ...")