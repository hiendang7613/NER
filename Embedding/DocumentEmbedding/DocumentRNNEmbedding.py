from Embedding.DocumentEmbedding.DocumentEmbedding import DocumentEmbedding

class DocumentRNNEmbeddings(DocumentEmbedding):
    def __init__(
        self,
        embeddings: List[TokenEmbeddings],
        hidden_size=128,
        rnn_layers=1,
        reproject_words: bool = True,
        reproject_words_dimension: int = None,
        bidirectional: bool = False,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
        rnn_type="GRU",
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
            )
        else:
            self.rnn = torch.nn.GRU(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
            )

        self.name = "document_" + self.rnn._get_name()

        # dropouts
        if locked_dropout > 0.0:
            self.dropout: torch.nn.Module = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        # the permutation that sorts the sentences by length, descending
        sort_perm = np.argsort([len(s) for s in sentences])[::-1]

        # the inverse permutation that restores the input order; it's an index tensor therefore LongTensor
        sort_invperm = np.argsort(sort_perm)

        # sort sentences by number of tokens
        sentences = [sentences[i] for i in sort_perm]

        self.embeddings.embed(sentences)

        longest_token_sequence_in_batch: int = len(sentences[0])

        # all_sentence_tensors = []
        lengths: List[int] = []

        # initialize zero-padded word embeddings tensor
        sentence_tensor = torch.zeros(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ],
            dtype=torch.float,
            device=flair.device,
        )

        # fill values with word embeddings
        for s_id, sentence in enumerate(sentences):
            lengths.append(len(sentence.tokens))

            sentence_tensor[s_id][: len(sentence)] = torch.cat(
                [token.get_embedding().unsqueeze(0) for token in sentence], 0
            )

        # TODO: this can only be removed once the implementations of word_dropout and locked_dropout have a batch_first mode
        sentence_tensor = sentence_tensor.transpose_(0, 1)

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        # use word dropout if set
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        sentence_tensor = self.dropout(sentence_tensor)
        packed = pack_padded_sequence(sentence_tensor, lengths)

        self.rnn.flatten_parameters()

        rnn_out, hidden = self.rnn(packed)

        outputs, output_lengths = pad_packed_sequence(rnn_out)

        outputs = self.dropout(outputs)

        # --------------------------------------------------------------------
        # EXTRACT EMBEDDINGS FROM RNN
        # --------------------------------------------------------------------
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[length - 1, sentence_no]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[0, sentence_no]
                embedding = torch.cat([first_rep, last_rep], 0)

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

        # restore original order of sentences in the batch
        sentences = [sentences[i] for i in sort_invperm]

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass
