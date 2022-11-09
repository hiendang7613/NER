from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class POSEmbeddings(TokenEmbedding):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        vocab: List = None,
        pos_embedding_dim: int = 50,
        debug = False,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "pos"
        self.static_embeddings = False
        self.debug = debug
        # use list of common characters if none provided
        self.pos_dictionary = {'unk': 0,'_': 1}
        for word in vocab:
            if word not in self.pos_dictionary:
                self.pos_dictionary[word] = len(self.pos_dictionary)


        self.pos_embedding_dim: int = pos_embedding_dim
        self.pos_embedding = torch.nn.Embedding(
            len(self.pos_dictionary), self.pos_embedding_dim
        )
        self.__embedding_length = self.pos_embedding_dim

    def _init_embeddings(self):
        utils.init_embedding(self.pos_embedding)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed_sentences(self, sentences):
        # pdb.set_trace()
        words=getattr(sentences,self.name).to(flair.device)
        embeddings = self.pos_embedding(words)
        return embeddings

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        embeddings = self.embed_sentences(sentences)
        embeddings = embeddings.cpu()
        sentences.features[self.name]=embeddings
        return sentences

    def __str__(self):
        return self.name
