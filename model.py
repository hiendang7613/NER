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

    @staticmethod
    def getModel(self, config):
        backbone_config = {}.update((k, config[k]) for k in ['backbone_type', 'input_shape', 'output_embedding_size'])
        embedding_config = {}.update((k, config[k]) for k in ['embedding_type', 'embedding_size'])
        head_config = {}.update((k, config[k]) for k in ['head_type', 'num_classes'])

        embedding, subword_tokenizer = EmbeddingFactory.getEmbedding(embedding_config)
        backbone_config['embedding'] = embedding
        backbone = BackboneFactory.getBackbone(backbone_config)
        head = HeadFactory.getHead(head_config)
        model = MyModel(backbone, head)
        return model, subword_tokenizer


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