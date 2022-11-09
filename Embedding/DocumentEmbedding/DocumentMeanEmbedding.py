from Embedding.DocumentEmbedding.DocumentEmbedding import DocumentEmbedding

class DocumentMeanEmbeddings(DocumentEmbedding):
    @deprecated(
        version="0.3.1",
        reason="The functionality of this class is moved to 'DocumentPoolEmbeddings'",
    )
    def __init__(self, token_embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings(
            embeddings=token_embeddings
        )
        self.name: str = "document_mean"

        self.__embedding_length: int = self.embeddings.embedding_length

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""

        everything_embedded: bool = True

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for sentence in sentences:
            if self.name not in sentence._embeddings.keys():
                everything_embedded = False

        if not everything_embedded:

            self.embeddings.embed(sentences)

            for sentence in sentences:
                word_embeddings = []
                for token in sentence.tokens:
                    word_embeddings.append(token.get_embedding().unsqueeze(0))

                word_embeddings = torch.cat(word_embeddings, dim=0).to(flair.device)

                mean_embedding = torch.mean(word_embeddings, 0)

                sentence.set_embedding(self.name, mean_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass
