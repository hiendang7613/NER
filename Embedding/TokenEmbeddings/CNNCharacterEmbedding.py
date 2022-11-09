from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class CNNCharacterEmbeddings(TokenEmbedding):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        vocab: List = None,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
        char_cnn = False,
        debug = False,
        embedding_name: str = None,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "Char"
        if embedding_name is not None:
            self.name = embedding_name
        self.static_embeddings = False
        self.char_cnn = char_cnn
        self.debug = debug
        # use list of common characters if none provided
        self.char_dictionary = {'<u>': 0}
        for word in vocab:
            for c in word:
                if c not in self.char_dictionary:
                    self.char_dictionary[c] = len(self.char_dictionary)

        self.char_dictionary[' '] = len(self.char_dictionary)  # concat for char
        self.char_dictionary['\n'] = len(self.char_dictionary)  # eof for char

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary), self.char_embedding_dim
        )
        if self.char_cnn:
            print("Use character level CNN")
            self.char_drop = torch.nn.Dropout(0.5)
            self.char_layer = torch.nn.Conv1d(char_embedding_dim, hidden_size_char, kernel_size=3, padding=1)
            self.__embedding_length = self.char_embedding_dim
        else:
            self.char_layer = torch.nn.LSTM(
                self.char_embedding_dim,
                self.hidden_size_char,
                num_layers=1,
                bidirectional=True,
            )

            self.__embedding_length = self.char_embedding_dim * 2
        self.pad_word = rand_emb(torch.FloatTensor(self.__embedding_length)).unsqueeze(0).unsqueeze(0)

    def _init_embeddings(self):
        utils.init_embedding(self.char_embedding)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed_sentences(self, sentences):
        # batch_size = len(sentences)
        # char_batch = sentences[0].total_len
        batch_size = len(sentences)
        char_batch = sentences.max_sent_len
        # char_list = []
        # char_lens = []
        # for sent in sentences:
        #     char_list.append(sent.char_id)
        #     char_lens.append(sent.char_lengths)
        # char_lengths = torch.cat(char_lens, 0)
        # char_seqs = pad_sequence(char_list, batch_first=False, padding_value=0)
        # char_seqs = char_seqs.view(-1, batch_size * char_batch)
        char_seqs = sentences.char_seqs.to(flair.device)
        char_lengths = sentences.char_lengths.to(flair.device)
        char_embeds = self.char_embedding(char_seqs)
        if self.char_cnn:
            char_embeds = self.char_drop(char_embeds)
            char_cnn_out = self.char_layer(char_embeds.permute(1,2,0))
            char_cnn_out = torch.nn.functional.max_pool1d(char_cnn_out, char_cnn_out.size(2)).view(batch_size, char_batch, -1)
            outs = char_cnn_out
        else:
            pack_char_seqs = pack_padded_sequence(input=char_embeds, lengths=char_lengths.to('cpu'), batch_first=False, enforce_sorted=False)
            lstm_out, hidden = self.char_layer(pack_char_seqs, None)
            # lstm_out = lstm_out.view(-1, self.hidden_dim)
            # hidden[0] = h_t = (2, b * s, 25)
            outs = hidden[0].transpose(0, 1).contiguous().view(batch_size * char_batch, -1)
            outs = outs.view(batch_size, char_batch, -1)

        return outs
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        embeddings = self.embed_sentences(sentences)
        embeddings = embeddings.cpu()
        sentences.features[self.name]=embeddings
        return sentences

    def __str__(self):
        return self.name