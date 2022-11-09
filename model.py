import tensorflow as tf
from Backbone.backbone import BackboneFactory
from Embedding.embedding import EmbeddingFactory
from Head.head import HeadFactory

class MyModel(tf.keras.Model):
    def __init__(self, backbone, head, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.backbone = backbone
        self.head = head

    def call(self, inputs, training=False):
        out = self.backbone(inputs, training=training)
        out = self.head(out, training=training)
        return out

class ModelFactory(object):
    def __init__(self):
        super(ModelFactory, self).__init__()
        pass

    '''
    ## getModel parameters
    modelConfig = {
        'backbone_type': backbone_type,
        'input_shape' : (width, height, chanels) 
        'output_embedding_size' : output_embedding_size # backbone output
        'embedding_type' :'phobert'
        'embedding_size' : embedding_size # backbone intput embedding 
        'head_type' : head_type,
        'num_classes' : num_classes
    }'''
    @staticmethod
    def getModel(self, config):
        backboneConfig={}
        backboneConfig.update((k, config[k]) for k in ['backbone_type', 'input_shape', 'output_embedding_size'])
        embeddingConfig={}
        embeddingConfig.update((k, config[k]) for k in ['embedding_type', 'embedding_size'])
        headConfig={}
        headConfig.update((k, config[k]) for k in ['head_type', 'num_classes'])

        embedding = EmbeddingFactory.getEmbedding(**embeddingConfig)
        backboneConfig['embedding'] = embedding
        backbone = BackboneFactory.getBackbone(**backboneConfig)
        head = HeadFactory.getHead(**headConfig)
        model = MyModel(backbone, head)
        return model


# if __name__ == '__main__':
    # """
    #     - ResNet_v1_101
    #     - ResNet_v1_34
    #     - Resnet_tf
    #     - Vgg16
    # """
    # input_shape = 250
    # model = MyModel(type_backbone='Resnet_tf',
    #                 input_shape=input_shape,
    #                 header=ArcHead(num_classes=1000, kernel_regularizer=tf.keras.regularizers.l2(5e-4)))
    # model.build(input_shape=(None, input_shape, input_shape, 3))
    # print(model.summary())
    #
    # x = tf.keras.layers.Input(shape=(input_shape, input_shape, 3))
    # out = model(x, training=True)
    #
    # print(f"input: {x}")
    # print(f"output: {out}")
    # print("DONE ...")