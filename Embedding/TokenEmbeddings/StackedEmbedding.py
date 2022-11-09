from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class StackedEmbeddings(TokenEmbedding):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings], gpu_friendly = False):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            self.add_module("list_embedding_{}".format(i), embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = True
        # self.gpu_friendly = gpu_friendly
        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True, embedding_mask = None
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]
        if embedding_mask is not None:
            # sort embeddings by name
            embedlist = sorted([(embedding.name, embedding) for embedding in self.embeddings], key = lambda x: x[0])
            for idx, embedding_tuple in enumerate(embedlist):
                embedding = embedding_tuple[1]
                if embedding_mask[idx] == 1:
                    # embedding.to(flair.device)
                    embedding.embed(sentences)
                    # embedding.to('cpu')
                else:
                    embedding.assign_batch_features(sentences, assign_zero=True)
        else:
            for embedding in self.embeddings:
                embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'