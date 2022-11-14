from transformers import AutoModel, AutoTokenizer
import tensorflow as tf


class EmbeddingFactory(object):
    def __init__(self):
        super(EmbeddingFactory, self).__init__()
        pass



    @staticmethod
    def getTokenizer(self, config):
        model_name = self.getRawModelName(config['embedding_type'])

        if 'ETNLP' in model_name:
            embedding = tf.saved_model.load('../Pretrained/Embeddings/'+model_name)
        else:
            subword_tokenizer = AutoTokenizer.from_pretrained(model_name)

        return subword_tokenizer

    @staticmethod
    def getEmbedding(self, config):
        model_name = self.getRawModelName(config['embedding_type'])

        if 'ETNLP' in model_name:
            embedding = tf.saved_model.load('../Pretrained/Embeddings/'+model_name)
        else:
            embedding = AutoModel.from_pretrained(model_name)

        return embedding


    @staticmethod
    def getRawModelName(self, embedding_type):

        # Vietnamese
        if embedding_type == 'bert-vi':
            model_name = "vinai/phobert-base"
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'trans-align-vi':
            model_name = 'Helsinki-NLP/opus-mt-vi-en'
        elif embedding_type == 'xlm-roberta-vi':
            model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
        elif embedding_type == 't5-vi':
            model_name = 'VietAI/vit5-base'

        # ETNLP - Pretrained Embedding
        # Training data: Wiki in Vietnamese -- sentences: 6,685,621 -- tokenized words: 114,997,587
        elif embedding_type == 'w2v-vi':
            model_name = 'W2V_ner_ETNLP'
        elif embedding_type == 'w2v-c2v-vi':
            model_name = 'W2V_C2V_ner_ETNLP'
        elif embedding_type == 'fasttext-vi':
            model_name = 'FastText_ner_ETNLP'
        elif embedding_type == 'elmo-vi':
            model_name = 'ELMO_ner_ETNLP'
        elif embedding_type == 'w2v-c2v-fasttext-elmo-bert-vi':
            model_name = 'MULTI_W_F_B_E_ETNLP'



        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'

        # English
        elif embedding_type == 'bert':
            model_name = "bert-base-uncased"


        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        elif embedding_type == 'gpt-vi':
            model_name = 'VietAI/gpt-neo-1.3B-vietnamese-news'
        else:
            print('Embedding type not match!')
# class Embedding(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(Head, self).__init__(**kwargs)
#         pass
#
#     def call(self, inputs, training=False) -> tf.Tensor:
#         pass
#
#
# class Embeddings(torch.nn.Module):
#     """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""
#
#     @property
#     @abstractmethod
#     def embedding_length(self) -> int:
#         """Returns the length of the embedding vector."""
#         pass
#
#     @property
#     @abstractmethod
#     def embedding_type(self) -> str:
#         pass
#
#     def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
#         """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
#         are non-static."""
#
#         # if only one sentence is passed, convert to list of sentence
#         if type(sentences) is Sentence:
#             sentences = [sentences]
#
#         everything_embedded: bool = True
#         if self.embedding_type == "word-level":
#             for sentence in sentences:
#                 for token in sentence.tokens:
#                     if self.name not in token._embeddings.keys():
#                         everything_embedded = False
#                         break
#                 if not everything_embedded:
#                     break
#         else:
#             for sentence in sentences:
#                 if self.name not in sentence._embeddings.keys():
#                     everything_embedded = False
#                     break
#
#         if not everything_embedded or not self.static_embeddings or (hasattr(sentences,'features') and self.name not in sentences.features):
#             self._add_embeddings_internal(sentences)
#
#         return sentences
#
#     @abstractmethod
#     def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
#         """Private method for adding embeddings to all words in a list of sentences."""
#         pass
#
#     def assign_batch_features(self, sentences, embedding_length=None, assign_zero=False):
#         if embedding_length is None:
#             embedding_length = self.embedding_length
#         sentence_lengths = [len(x) for x in sentences]
#         if not assign_zero:
#             sentence_tensor = torch.zeros([len(sentences),max(sentence_lengths),embedding_length]).type_as(sentences[0][0]._embeddings[self.name])
#             for sent_id, sentence in enumerate(sentences):
#                 for token_id, token in enumerate(sentence):
#                     sentence_tensor[sent_id,token_id]=token._embeddings[self.name]
#         else:
#             sentence_tensor = torch.zeros([len(sentences),max(sentence_lengths),embedding_length]).float()
#         sentence_tensor = sentence_tensor.cpu()
#         sentences.features[self.name]=sentence_tensor
#         return sentences



