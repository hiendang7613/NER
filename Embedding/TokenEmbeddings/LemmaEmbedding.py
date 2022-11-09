from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class LemmaEmbeddings(TokenEmbedding):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        vocab: List = None,
        lemma_embedding_dim: int = 100,
        debug = False,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "lemma"
        self.static_embeddings = False
        self.debug = debug
        # use list of common characters if none provided
        self.lemma_dictionary = {'unk': 0,'_': 1}
        for word in vocab:
            if word not in self.lemma_dictionary:
                self.lemma_dictionary[word] = len(self.lemma_dictionary)


        self.lemma_embedding_dim: int = lemma_embedding_dim
        self.lemma_embedding = torch.nn.Embedding(
            len(self.lemma_dictionary), self.lemma_embedding_dim
        )
        self.__embedding_length = self.lemma_embedding_dim

    def _init_embeddings(self):
        utils.init_embedding(self.lemma_embedding)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed_sentences(self, sentences):
        # pdb.set_trace()
        words=getattr(sentences,self.name).to(flair.device)
        embeddings = self.lemma_embedding(words)
        return embeddings

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        embeddings = self.embed_sentences(sentences)
        embeddings = embeddings.cpu()
        sentences.features[self.name]=embeddings
        return sentences

    def __str__(self):
        return self.name
