from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class ELMoEmbeddings(TokenEmbedding):
    """Contextual word embeddings using word-level LM, as proposed in Peters et al., 2018."""

    def __init__(
            self, model: str = "original", options_file: str = None, weight_file: str = None, embedding_name=None,
    ):
        super().__init__()

        try:
            import allennlp.commands.elmo
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                'To use ELMoEmbeddings, please first install with "pip install allennlp"'
            )
            log.warning("-" * 100)
            pass

        if embedding_name is None:
            self.name = "elmo-" + model
        else:
            self.name = embedding_name

        self.static_embeddings = True

        if not options_file or not weight_file:
            # the default model for ELMo is the 'original' model, which is very large
            options_file = allennlp.commands.elmo.DEFAULT_OPTIONS_FILE
            weight_file = allennlp.commands.elmo.DEFAULT_WEIGHT_FILE
            # alternatively, a small, medium or portuguese model can be selected by passing the appropriate mode name
            if model == "small":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
            if model == "medium":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            if model in ["large", "5.5B"]:
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
            if model == "pt" or model == "portuguese":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pt/elmo_pt_weights.hdf5"
            if model == "pubmed":
                options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"

        # put on Cuda if available
        from flair import device

        if re.fullmatch(r"cuda:[0-9]+", str(device)):
            cuda_device = int(str(device).split(":")[-1])
        elif str(device) == "cpu":
            cuda_device = -1
        else:
            cuda_device = 0

        self.options_file = options_file
        self.weight_file = weight_file
        self.cuda_device = cuda_device
        self.ee = allennlp.commands.elmo.ElmoEmbedder(
            options_file=options_file, weight_file=weight_file, cuda_device=cuda_device
        )

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

    def reset_elmo(self):
        try:
            import allennlp.commands.elmo
        except ModuleNotFoundError:
            log.warning("-" * 100)
            log.warning('ATTENTION! The library "allennlp" is not installed!')
            log.warning(
                'To use ELMoEmbeddings, please first install with "pip install allennlp"'
            )
            log.warning("-" * 100)
            pass

        self.ee = allennlp.commands.elmo.ElmoEmbedder(
            options_file=self.options_file, weight_file=self.weight_file, cuda_device=self.cuda_device
        )

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        if hasattr(sentences, 'features'):
            if self.name in sentences.features:
                return sentences
            if len(sentences) > 0:
                if self.name in sentences[0][0]._embeddings.keys():
                    sentences = self.assign_batch_features(sentences)
                    return sentences
        sentence_words: List[List[str]] = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])
        # pdb.set_trace()
        embeddings = self.ee.embed_batch(sentence_words)

        for i, sentence in enumerate(sentences):

            sentence_embeddings = embeddings[i]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embedding = torch.cat(
                    [
                        torch.FloatTensor(sentence_embeddings[0, token_idx, :]),
                        torch.FloatTensor(sentence_embeddings[1, token_idx, :]),
                        torch.FloatTensor(sentence_embeddings[2, token_idx, :]),
                    ],
                    0,
                )

                token.set_embedding(self.name, word_embedding)
        if hasattr(sentences, 'features'):
            sentences = self.assign_batch_features(sentences)
        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name
