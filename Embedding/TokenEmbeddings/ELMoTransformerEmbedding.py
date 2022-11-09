from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class ELMoTransformerEmbeddings(TokenEmbedding):
    """Contextual word embeddings using word-level Transformer-based LM, as proposed in Peters et al., 2018."""

    @deprecated(
        version="0.4.2",
        reason="Not possible to load or save ELMo Transformer models. @stefan-it is working on it.",
    )
    def __init__(self, model_file: str):
        super().__init__()

        try:
            from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import (
                BidirectionalLanguageModelTokenEmbedder,
            )
            from allennlp.data.token_indexers.elmo_indexer import (
                ELMoTokenCharactersIndexer,
            )
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                "To use ELMoTransformerEmbeddings, please first install a recent version from https://github.com/allenai/allennlp"
            )
            log.warning("-" * 100)
            pass

        self.name = "elmo-transformer"
        self.static_embeddings = True
        self.lm_embedder = BidirectionalLanguageModelTokenEmbedder(
            archive_file=model_file,
            dropout=0.2,
            bos_eos_tokens=("<S>", "</S>"),
            remove_bos_eos=True,
            requires_grad=False,
        )
        self.lm_embedder = self.lm_embedder.to(device=flair.device)
        self.vocab = self.lm_embedder._lm.vocab
        self.indexer = ELMoTokenCharactersIndexer()

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        # Avoid conflicts with flair's Token class
        import allennlp.data.tokenizers.token as allen_nlp_token

        indexer = self.indexer
        vocab = self.vocab
        with torch.no_grad():
            for sentence in sentences:
                character_indices = indexer.tokens_to_indices(
                    [allen_nlp_token.Token(token.text) for token in sentence], vocab, "elmo"
                )["elmo"]

                indices_tensor = torch.LongTensor([character_indices])
                indices_tensor = indices_tensor.to(device=flair.device)
                embeddings = self.lm_embedder(indices_tensor)[0].detach().cpu().numpy()

                for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                    embedding = embeddings[token_idx]
                    word_embedding = torch.FloatTensor(embedding)
                    token.set_embedding(self.name, word_embedding)

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name

