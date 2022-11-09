from Embedding.DocumentEmbedding.DocumentEmbedding import DocumentEmbedding

class DocumentLMEmbeddings(DocumentEmbedding):
    def __init__(self, flair_embeddings: List[FlairEmbeddings]):
        super().__init__()

        self.embeddings = flair_embeddings
        self.name = "document_lm"

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(flair_embeddings):
            self.add_module("lm_embedding_{}".format(i), embedding)
            if not embedding.static_embeddings:
                self.static_embeddings = False

        self._embedding_length: int = sum(
            embedding.embedding_length for embedding in flair_embeddings
        )

    @property
    def embedding_length(self) -> int:
        return self._embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

            # iterate over sentences
            for sentence in sentences:
                sentence: Sentence = sentence

                # if its a forward LM, take last state
                if embedding.is_forward_lm:
                    sentence.set_embedding(
                        embedding.name,
                        sentence[len(sentence) - 1]._embeddings[embedding.name],
                    )
                else:
                    sentence.set_embedding(
                        embedding.name, sentence[0]._embeddings[embedding.name]
                    )

        return sentences

