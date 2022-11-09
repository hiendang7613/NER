from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class OpenAIGPTEmbeddings(TokenEmbedding):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai-gpt",
        layers: str = "1",
        pooling_operation: str = "first_last",
        use_scalar_mix: bool = False,
    ):
        """OpenAI GPT embeddings, as proposed in Radford et al. 2018.
        :param pretrained_model_name_or_path: name or path of OpenAI GPT model
        :param layers: comma-separated list of layers
        :param pooling_operation: defines pooling operation for subwords
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()

        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.model = OpenAIGPTModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
        )
        self.name = pretrained_model_name_or_path
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.static_embeddings = True

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
        self.model.to(flair.device)
        self.model.eval()

        sentences = _get_transformer_sentence_embeddings(
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.model,
            name=self.name,
            layers=self.layers,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix,
        )

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name

