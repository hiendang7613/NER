from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class FastTextEmbeddings(TokenEmbedding):
    """FastText Embeddings with oov functionality"""

    def __init__(self, embeddings: str, use_local: bool = True, field: str = None):
        """
        Initializes fasttext word embeddings. Constructor downloads required embedding file and stores in cache
        if use_local is False.

        :param embeddings: path to your embeddings '.bin' file
        :param use_local: set this to False if you are using embeddings from a remote source
        """

        cache_dir = Path("embeddings")

        if use_local:
            if not Path(embeddings).exists():
                raise ValueError(
                    f'The given embeddings "{embeddings}" is not available or is not a valid path.'
                )
        else:
            embeddings = cached_path(f"{embeddings}", cache_dir=cache_dir)

        self.embeddings = embeddings

        self.name: str = str(embeddings)

        self.static_embeddings = True

        self.precomputed_word_embeddings = gensim.models.FastText.load_fasttext_format(
            str(embeddings)
        )

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size

        self.field = field
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                try:
                    word_embedding = self.precomputed_word_embeddings[word]
                except:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return f"'{self.embeddings}'"
