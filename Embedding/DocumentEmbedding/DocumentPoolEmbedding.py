from Embedding.DocumentEmbedding.DocumentEmbedding import DocumentEmbedding

class DocumentPoolEmbeddings(DocumentEmbedding):
    def __init__(
        self,
        embeddings: List[TokenEmbeddings],
        fine_tune_mode="linear",
        pooling: str = "mean",
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param pooling: a string which can any value from ['mean', 'max', 'min']
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length

        # optional fine-tuning on top of embedding layer
        self.fine_tune_mode = fine_tune_mode
        if self.fine_tune_mode in ["nonlinear", "linear"]:
            self.embedding_flex = torch.nn.Linear(
                self.embedding_length, self.embedding_length, bias=False
            )
            self.embedding_flex.weight.data.copy_(torch.eye(self.embedding_length))

        if self.fine_tune_mode in ["nonlinear"]:
            self.embedding_flex_nonlinear = torch.nn.ReLU(self.embedding_length)
            self.embedding_flex_nonlinear_map = torch.nn.Linear(
                self.embedding_length, self.embedding_length
            )

        self.__embedding_length: int = self.embeddings.embedding_length

        self.to(flair.device)

        self.pooling = pooling
        if self.pooling == "mean":
            self.pool_op = torch.mean
        elif pooling == "max":
            self.pool_op = torch.max
        elif pooling == "min":
            self.pool_op = torch.min
        else:
            raise ValueError(f"Pooling operation for {self.mode!r} is not defined")
        self.name: str = f"document_{self.pooling}"

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        self.embeddings.embed(sentences)

        for sentence in sentences:
            word_embeddings = []
            for token in sentence.tokens:
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            word_embeddings = torch.cat(word_embeddings, dim=0).to(flair.device)

            if self.fine_tune_mode in ["nonlinear", "linear"]:
                word_embeddings = self.embedding_flex(word_embeddings)

            if self.fine_tune_mode in ["nonlinear"]:
                word_embeddings = self.embedding_flex_nonlinear(word_embeddings)
                word_embeddings = self.embedding_flex_nonlinear_map(word_embeddings)

            if self.pooling == "mean":
                pooled_embedding = self.pool_op(word_embeddings, 0)
            else:
                pooled_embedding, _ = self.pool_op(word_embeddings, 0)

            sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass

    def extra_repr(self):
        return f"fine_tune_mode={self.fine_tune_mode}, pooling={self.pooling}"

