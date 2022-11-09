from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding


class TransformerXLEmbeddings(TokenEmbedding):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "transfo-xl-wt103",
        layers: str = "1,2,3",
        use_scalar_mix: bool = False,
    ):
        """Transformer-XL embeddings, as proposed in Dai et al., 2019.
        :param pretrained_model_name_or_path: name or path of Transformer-XL model
        :param layers: comma-separated list of layers
        :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
        """
        super().__init__()

        self.tokenizer = TransfoXLTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        self.model = TransfoXLModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
        )
        self.name = pretrained_model_name_or_path
        self.layers: List[int] = [int(layer) for layer in layers.split(",")]
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
            pooling_operation="first",
            use_scalar_mix=self.use_scalar_mix,
            eos_token="<eos>",
        )
        return sentences

    def extra_repr(self):
        return "model={}".format(self.name)

    def __str__(self):
        return self.name


def _extract_embeddings(
    hidden_states: List[torch.FloatTensor],
    layers: List[int],
    pooling_operation: str,
    subword_start_idx: int,
    subword_end_idx: int,
    use_scalar_mix: bool = False,
) -> List[torch.FloatTensor]:
    """
    Extracts subword embeddings from specified layers from hidden states.
    :param hidden_states: list of hidden states from model
    :param layers: list of layers
    :param pooling_operation: pooling operation for subword embeddings (supported: first, last, first_last and mean)
    :param subword_start_idx: defines start index for subword
    :param subword_end_idx: defines end index for subword
    :param use_scalar_mix: determines, if scalar mix should be used
    :return: list of extracted subword embeddings
    """
    subtoken_embeddings: List[torch.FloatTensor] = []

    for layer in layers:
        current_embeddings = hidden_states[layer][0][subword_start_idx:subword_end_idx]

        first_embedding: torch.FloatTensor = current_embeddings[0]
        if pooling_operation == "first_last":
            last_embedding: torch.FloatTensor = current_embeddings[-1]
            final_embedding: torch.FloatTensor = torch.cat(
                [first_embedding, last_embedding]
            )
        elif pooling_operation == "last":
            final_embedding: torch.FloatTensor = current_embeddings[-1]
        elif pooling_operation == "mean":
            all_embeddings: List[torch.FloatTensor] = [
                embedding.unsqueeze(0) for embedding in current_embeddings
            ]
            final_embedding: torch.FloatTensor = torch.mean(
                torch.cat(all_embeddings, dim=0), dim=0
            )
        else:
            final_embedding: torch.FloatTensor = first_embedding

        subtoken_embeddings.append(final_embedding)

    if use_scalar_mix:
        sm = ScalarMix(mixture_size=len(subtoken_embeddings))
        sm_embeddings = sm(subtoken_embeddings)

        subtoken_embeddings = [sm_embeddings]

    return subtoken_embeddings


def _build_token_subwords_mapping(
    sentence: Sentence, tokenizer: PreTrainedTokenizer
) -> Dict[int, int]:
    """ Builds a dictionary that stores the following information:
    Token index (key) and number of corresponding subwords (value) for a sentence.

    :param sentence: input sentence
    :param tokenizer: PyTorch-Transformers tokenization object
    :return: dictionary of token index to corresponding number of subwords
    """
    token_subwords_mapping: Dict[int, int] = {}

    for token in sentence.tokens:
        token_text = token.text

        subwords = tokenizer.tokenize(token_text)

        token_subwords_mapping[token.idx] = len(subwords)

    return token_subwords_mapping


def _build_token_subwords_mapping_gpt2(
    sentence: Sentence, tokenizer: PreTrainedTokenizer
) -> Dict[int, int]:
    """ Builds a dictionary that stores the following information:
    Token index (key) and number of corresponding subwords (value) for a sentence.

    :param sentence: input sentence
    :param tokenizer: PyTorch-Transformers tokenization object
    :return: dictionary of token index to corresponding number of subwords
    """
    token_subwords_mapping: Dict[int, int] = {}

    for token in sentence.tokens:
        # Dummy token is needed to get the actually token tokenized correctly with special ``Ġ`` symbol

        if token.idx == 1:
            token_text = token.text
            subwords = tokenizer.tokenize(token_text)
        else:
            token_text = "X " + token.text
            subwords = tokenizer.tokenize(token_text)[1:]

        token_subwords_mapping[token.idx] = len(subwords)

    return token_subwords_mapping


def _get_transformer_sentence_embeddings(
    sentences: List[Sentence],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    name: str,
    layers: List[int],
    pooling_operation: str,
    use_scalar_mix: bool,
    bos_token: str = None,
    eos_token: str = None,
    gradient_context = torch.no_grad(),
) -> List[Sentence]:
    """
    Builds sentence embeddings for Transformer-based architectures.
    :param sentences: input sentences
    :param tokenizer: tokenization object
    :param model: model object
    :param name: name of the Transformer-based model
    :param layers: list of layers
    :param pooling_operation: defines pooling operation for subword extraction
    :param use_scalar_mix: defines the usage of scalar mix for specified layer(s)
    :param bos_token: defines begin of sentence token (used for left padding)
    :param eos_token: defines end of sentence token (used for right padding)
    :return: list of sentences (each token of a sentence is now embedded)
    """
    with gradient_context:
        for sentence in sentences:
            token_subwords_mapping: Dict[int, int] = {}

            if name.startswith("gpt2") or name.startswith("roberta"):
                token_subwords_mapping = _build_token_subwords_mapping_gpt2(
                    sentence=sentence, tokenizer=tokenizer
                )
            else:
                token_subwords_mapping = _build_token_subwords_mapping(
                    sentence=sentence, tokenizer=tokenizer
                )
            subwords = tokenizer.tokenize(sentence.to_tokenized_string())

            offset = 0

            if 'xlmr' in name:
                offset = 1

            if bos_token:
                subwords = [bos_token] + subwords
                offset = 1

            if eos_token:
                subwords = subwords + [eos_token]

            indexed_tokens = tokenizer.convert_tokens_to_ids(subwords)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(flair.device)

            hidden_states = model(tokens_tensor)[-1]

            for token in sentence.tokens:
                len_subwords = token_subwords_mapping[token.idx]

                subtoken_embeddings = _extract_embeddings(
                    hidden_states=hidden_states,
                    layers=layers,
                    pooling_operation=pooling_operation,
                    subword_start_idx=offset,
                    subword_end_idx=offset + len_subwords,
                    use_scalar_mix=use_scalar_mix,
                )

                offset += len_subwords

                final_subtoken_embedding = torch.cat(subtoken_embeddings)
                token.set_embedding(name, final_subtoken_embedding)

    return sentences
