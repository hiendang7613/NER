import tensorflow as tf

from Backbone.RNN import BiLSTM

class Embedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Head, self).__init__(**kwargs)
        pass

    def call(self, inputs, training=False) -> tf.Tensor:
        pass


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True
        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
                        break
                if not everything_embedded:
                    break
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False
                    break

        if not everything_embedded or not self.static_embeddings or (hasattr(sentences,'features') and self.name not in sentences.features):
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass

    def assign_batch_features(self, sentences, embedding_length=None, assign_zero=False):
        if embedding_length is None:
            embedding_length = self.embedding_length
        sentence_lengths = [len(x) for x in sentences]
        if not assign_zero:
            sentence_tensor = torch.zeros([len(sentences),max(sentence_lengths),embedding_length]).type_as(sentences[0][0]._embeddings[self.name])
            for sent_id, sentence in enumerate(sentences):
                for token_id, token in enumerate(sentence):
                    sentence_tensor[sent_id,token_id]=token._embeddings[self.name]
        else:
            sentence_tensor = torch.zeros([len(sentences),max(sentence_lengths),embedding_length]).float()
        sentence_tensor = sentence_tensor.cpu()
        sentences.features[self.name]=sentence_tensor
        return sentences


class EmbeddingFactory(object):
    def __init__(self):
        super(EmbeddingFactory, self).__init__()
        pass

    '''
    ## getHead parameters
    config = {
        'head_type' : head_type
        'num_classes' : num_classes
    }'''
    @staticmethod
    def getEmbedding(self, config):
        embedding = None
        # if embedding_type == 'head_1':  ### embedding_size
        #     head = BiLSTM(include_top=False)
        # elif backbone_type == 'backbone_1':
        #     output = BiLSTM(include_top=False)(input_tensor)
        # else:
        #     print('Head type not match!')
        return embedding


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