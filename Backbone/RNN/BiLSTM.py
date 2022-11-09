from cgi import print_environ
from unicodedata import name
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense

class BiLSTM(tf.keras.Model):
    def __init__(self, embedding_size, word_dropout_rate, dict_len):
        super(BiLSTM, self).__init__()
        self.word_dropout = Dropout(word_dropout_rate, input_shape=(-1, 1, 1))
        self.lstm1 = Bidirectional(LSTM(embedding_size, kernel_initializer='glorot_normal', return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(embedding_size, kernel_initializer='glorot_normal'))
        self.dense = Dense(dict_len)

    def call(self, inputs, training=False):
        x = self.word_dropout(inputs, training=training)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dense(x, training=training)

        return x




























####
if not self.new_drop:
    if dropout > 0.0:
        self.dropout = torch.nn.Dropout(dropout)

    if word_dropout > 0.0:
        self.word_dropout = flair.nn.WordDropout(word_dropout)

    if locked_dropout > 0.0:
        self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
else:
    self.dropout1 = torch.nn.Dropout(p=dropout)
    self.dropout2 = torch.nn.Dropout(p=dropout)
#####
sentence_tensor = self.dropout1(sentence_tensor)

if self.relearn_embeddings:
    sentence_tensor = self.embedding2nn(sentence_tensor)

#####
if self.use_rnn:
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        sentence_tensor, lengths, enforce_sorted=False
    )

    # if initial hidden state is trainable, use this state
    if self.train_initial_hidden_state:
        initial_hidden_state = [
            self.lstm_init_h.unsqueeze(1).repeat(1, len(sentences), 1),
            self.lstm_init_c.unsqueeze(1).repeat(1, len(sentences), 1),
        ]
        rnn_output, hidden = self.rnn(packed, initial_hidden_state)
    else:
        rnn_output, hidden = self.rnn(packed)
    sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    #dropout
    if not self.new_drop:
        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        # word dropout only before RNN - TODO: more experimentation needed
        # if self.use_word_dropout > 0.0:
        #     sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)






