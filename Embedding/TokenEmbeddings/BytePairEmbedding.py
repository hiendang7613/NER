from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class BytePairEmbeddings(TokenEmbedding):
    def __init__(
        self,
        language: str,
        dim: int = 50,
        syllables: int = 100000,
        cache_dir=Path(flair.cache_root) / "embeddings",
    ):
        """
        Initializes BP embeddings. Constructor downloads required files if not there.
        """

        self.name: str = f"bpe-{language}-{syllables}-{dim}"
        self.static_embeddings = True
        self.embedder = BPEmbSerializable(
            lang=language, vs=syllables, dim=dim, cache_dir=cache_dir
        )

        self.__embedding_length: int = self.embedder.emb.vector_size * 2
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

                if word.strip() == "":
                    # empty words get no embedding
                    token.set_embedding(
                        self.name, torch.zeros(self.embedding_length, dtype=torch.float)
                    )
                else:
                    # all other words get embedded
                    embeddings = self.embedder.embed(word.lower())
                    embedding = np.concatenate(
                        (embeddings[0], embeddings[len(embeddings) - 1])
                    )
                    token.set_embedding(
                        self.name, torch.tensor(embedding, dtype=torch.float)
                    )

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        return "model={}".format(self.name)




class BPEmbSerializable(BPEmb):
    def __getstate__(self):
        state = self.__dict__.copy()
        # save the sentence piece model as binary file (not as path which may change)
        state["spm_model_binary"] = open(self.model_file, mode="rb").read()
        state["spm"] = None
        return state

    def __setstate__(self, state):
        from bpemb.util import sentencepiece_load

        model_file = self.model_tpl.format(lang=state["lang"], vs=state["vs"])
        self.__dict__ = state

        # write out the binary sentence piece model into the expected directory
        self.cache_dir: Path = Path(flair.cache_root) / "embeddings"
        if "spm_model_binary" in self.__dict__:
            # if the model was saved as binary and it is not found on disk, write to appropriate path
            if not os.path.exists(self.cache_dir / state["lang"]):
                os.makedirs(self.cache_dir / state["lang"])
            self.model_file = self.cache_dir / model_file
            with open(self.model_file, "wb") as out:
                out.write(self.__dict__["spm_model_binary"])
        else:
            # otherwise, use normal process and potentially trigger another download
            self.model_file = self._load_file(model_file)

        # once the modes if there, load it with sentence piece
        state["spm"] = sentencepiece_load(self.model_file)

