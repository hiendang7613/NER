from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class MuseCrosslingualEmbeddings(TokenEmbedding):
    def __init__(self,):
        self.name: str = f"muse-crosslingual"
        self.static_embeddings = True
        self.__embedding_length: int = 300
        self.language_embeddings = {}
        super().__init__()

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            language_code = sentence.get_language_code()
            print(language_code)
            supported = [
                "en",
                "de",
                "bg",
                "ca",
                "hr",
                "cs",
                "da",
                "nl",
                "et",
                "fi",
                "fr",
                "el",
                "he",
                "hu",
                "id",
                "it",
                "mk",
                "no",
                "pl",
                "pt",
                "ro",
                "ru",
                "sk",
            ]
            if language_code not in supported:
                language_code = "en"

            if language_code not in self.language_embeddings:
                log.info(f"Loading up MUSE embeddings for '{language_code}'!")
                # download if necessary
                webpath = "https://alan-nlp.s3.eu-central-1.amazonaws.com/resources/embeddings-muse"
                cache_dir = Path("embeddings") / "MUSE"
                cached_path(
                    f"{webpath}/muse.{language_code}.vec.gensim.vectors.npy",
                    cache_dir=cache_dir,
                )
                embeddings_file = cached_path(
                    f"{webpath}/muse.{language_code}.vec.gensim", cache_dir=cache_dir
                )

                # load the model
                self.language_embeddings[
                    language_code
                ] = gensim.models.KeyedVectors.load(str(embeddings_file))

            current_embedding_model = self.language_embeddings[language_code]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                if word in current_embedding_model:
                    word_embedding = current_embedding_model[word]
                elif word.lower() in current_embedding_model:
                    word_embedding = current_embedding_model[word.lower()]
                elif re.sub(r"\d", "#", word.lower()) in current_embedding_model:
                    word_embedding = current_embedding_model[
                        re.sub(r"\d", "#", word.lower())
                    ]
                elif re.sub(r"\d", "0", word.lower()) in current_embedding_model:
                    word_embedding = current_embedding_model[
                        re.sub(r"\d", "0", word.lower())
                    ]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype="float")

                word_embedding = torch.FloatTensor(word_embedding)

                token.set_embedding(self.name, word_embedding)

        return sentences

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def __str__(self):
        return self.name
