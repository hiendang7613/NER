from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class XLMRoBERTaEmbeddings(TokenEmbedding):
    def __init__(
            self,
            pretrained_model_name_or_path: str = "xlm-roberta-large",
            layers: str = "-1",
            pooling_operation: str = "first",
            fine_tune: bool = False,
            use_scalar_mix: bool = False,
    ):
        """RoBERTa, as proposed by Liu et al. 2019.
        :param pretrained_model_name_or_path: name or path of RoBERTa model
        :param layers: comma-separated list of layers
        :param pooling_operation: defines pooling operation for subwords
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = XLMRobertaModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
        )

        self.name = pretrained_model_name_or_path
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix

        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        self.to(flair.device)
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        if self.static_embeddings:
            self.model.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        if not hasattr(self, 'fine_tune'):
            self.fine_tune = False
        if hasattr(sentences, 'features'):
            if not self.fine_tune:
                if self.name in sentences.features:
                    return sentences
                if len(sentences) > 0:
                    if self.name in sentences[0][0]._embeddings.keys():
                        sentences = self.assign_batch_features(sentences)
                        return sentences

        # self.model.to(flair.device)
        if not self.fine_tune:
            self.model.eval()
        else:
            self.model.train()

        gradient_context = torch.enable_grad() if self.fine_tune and self.training else torch.no_grad()
        sentences = _get_transformer_sentence_embeddings(
            sentences=sentences,
            tokenizer=self.tokenizer,
            model=self.model,
            name=self.name,
            layers=self.layers,
            pooling_operation=self.pooling_operation,
            use_scalar_mix=self.use_scalar_mix,
            bos_token="<s>",
            eos_token="</s>",
            gradient_context=gradient_context,
        )
        # pdb.set_trace()
        if hasattr(sentences, 'features'):
            sentences = self.assign_batch_features(sentences)

        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name

