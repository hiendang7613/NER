from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding


class RNNCharacterEmbeddings(TokenEmbedding):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        path_to_char_dict: str = None,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "Char"
        self.static_embeddings = False

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(
                path_to_char_dict
            )

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim
        )
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.__embedding_length = self.char_embedding_dim * 2

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for sentence in sentences:

            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                char_indices = [
                    self.char_dictionary.get_idx_for_item(char) for char in token.text
                ]
                tokens_char_indices.append(char_indices)

            # sort words by length, for batching and masking
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=lambda p: len(p), reverse=True
            )
            d = {}
            for i, ci in enumerate(tokens_char_indices):
                for j, cj in enumerate(tokens_sorted_by_length):
                    if ci == cj:
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence),
                dtype=torch.long,
                device=flair.device,
            )

            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[i, : chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device
                )

            # chars for rnn processing
            chars = tokens_mask

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length
            )

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)),
                dtype=torch.float,
                device=flair.device,
            )
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name

