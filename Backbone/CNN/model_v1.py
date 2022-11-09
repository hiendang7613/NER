import tensorflow as tf

elif self.use_cnn:

    # transpose to batch_first mode
    sentence_tensor = sentence_tensor.transpose_(0, 1)
    batch_size = len(sentences)
    word_in = torch.tanh(self.word2cnn(sentence_tensor)).transpose(2, 1).contiguous()
    for idx in range(self.nlayers):
        if idx == 0:
            cnn_feature = F.relu(self.cnn_list[idx](word_in))
        else:
            cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
        cnn_feature = self.cnn_drop_list[idx](cnn_feature)
        if batch_size > 1:
            cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
    sentence_tensor = cnn_feature.transpose(2, 1).contiguous()
else:
    # transpose to batch_first mode
    sentence_tensor = sentence_tensor.transpose_(0, 1)


from cgi import print_environ
from unicodedata import name
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, Dropout, BatchNormalization, Dense

class Block(tf.keras.layers.Layer):
    def __init__(self, embedding_size, dropout_rate=0.5):
        super(Block, self).__init__()
        self.conv = Conv1D(embedding_size, 3, padding='same')
        self.act = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.bn = BatchNormalization()
        pass
    def call(self, inputs, training=False):
        x = self.word_dropout(inputs, training=training)
        x = self.conv(x, training=training)
        x = self.act(x, training=training)
        x = self.dropout(x, training=training)
        x = self.bn(x, training=training)
        x = self.dropout(x, training=training)

        return x
class CNN(tf.keras.Model):
    def __init__(self, embedding_size, word_dropout_rate, dropout_rate=0.5, dict_len):
        super(CNN, self).__init__()
        self.word_dropout = Dropout(word_dropout_rate, input_shape=(-1, 1, 1))
        self.block1 = Block(embedding_size)
        self.block2 = Block(embedding_size)
        self.block3 = Block(embedding_size)
        # self.dense = Dense(embedding_size)
        # self.dense = Dense(embedding_size)
        self.dense = Dense(dict_len)

    def call(self, inputs, training=False):
        x = self.word_dropout(inputs, training=training)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.dense(x, training=training)
        return x
