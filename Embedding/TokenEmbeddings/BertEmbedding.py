from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class BertEmbeddings(TokenEmbedding):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
        fine_tune: bool = False,
        sentence_feat: bool = False,
        max_sequence_length = 510,
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path=bert_model_or_path, output_hidden_states=True)
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        if self.static_embeddings:
            self.model.eval()  # disable dropout (or leave in train mode to finetune)
        # if True, return the sentence_feat
        self.sentence_feat=sentence_feat
        self.max_sequence_length = max_sequence_length + 2

    # def _convert_sentences_to_features(
    #     self, sentences, max_sequence_length: int
    # ) -> [BertInputFeatures]:
    def __call__(self, sentences):

        token_ids_arr = []
        marks_arr = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization = []
            token_subtoken_count = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token.text)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[token.idx] = len(subtokens)
            if len(bert_tokenization) > self.max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0 : (self.max_sequence_length - 2)]

            tokens = []
            tokens.append("[CLS]")
            for token in bert_tokenization:
                tokens.append(token)
            tokens.append("[SEP]")
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding 0 value in the mask
            marks = [1] * len(token_ids)
            while len(token_ids) < self.max_sequence_length:
                token_ids.append(0)
                marks.append(0)

            token_ids_arr.append(token_ids)
            marks_arr.append(marks)

        return (token_ids_arr, marks_arr)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""
        if not hasattr(self,'fine_tune'):
            self.fine_tune=False
        if hasattr(sentences, 'features'):
            if not self.fine_tune:
                if self.name in sentences.features:
                    return sentences
                if len(sentences)>0:
                    if self.name in sentences[0][0]._embeddings.keys():
                        sentences = self.assign_batch_features(sentences)
                        return sentences

        # first, find longest sentence in batch
        try:
            longest_sentence_in_batch: int = len(
                max(
                    [
                        self.tokenizer.tokenize(sentence.to_tokenized_string())
                        for sentence in sentences
                    ],
                    key=len,
                )
            )
        except:
            pdb.set_trace()
        if not hasattr(self,'max_sequence_length'):
            self.max_sequence_length=510
        if longest_sentence_in_batch>self.max_sequence_length:
            longest_sentence_in_batch=self.max_sequence_length
        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(
            flair.device
        )
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).to(
            flair.device
        )


        # put encoded batch through BERT model to get all hidden states of all encoder layers
        # self.model.to(flair.device)
        # self.model.eval()
        if not self.fine_tune:
            self.model.eval()
        # pdb.set_trace()
        gradient_context = torch.enable_grad() if self.fine_tune and self.training else torch.no_grad()


        with gradient_context:
            sequence_output, pooled_output, all_encoder_layers = self.model(all_input_ids, token_type_ids=None, attention_mask=all_input_masks)
            # gradients are enable if fine-tuning is enabled
            if not hasattr(self,'sentence_feat'):
                self.sentence_feat=False
            if self.sentence_feat:
                self.pooled_output=pooled_output

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:
                        if self.use_scalar_mix:
                            layer_output = all_encoder_layers[int(layer_index)][
                                sentence_index
                            ]
                        else:
                            if not self.fine_tune:
                                layer_output = (
                                    all_encoder_layers[int(layer_index)]
                                    .detach()
                                    .cpu()[sentence_index]
                                    )
                            else:
                                layer_output = (
                                    all_encoder_layers[int(layer_index)][sentence_index]
                                    )

                        all_layers.append(layer_output[token_index])

                    if self.use_scalar_mix:
                        sm = ScalarMix(mixture_size=len(all_layers))
                        sm_embeddings = sm(all_layers)
                        all_layers = [sm_embeddings]

                    subtoken_embeddings.append(torch.cat(all_layers))

                # get the current sentence object
                token_idx = 0
                for posidx, token in enumerate(sentence):
                    # add concatenated embedding to sentence
                    token_idx += 1

                    if self.pooling_operation == "first":
                        # use first subword embedding if pooling operation is 'first'
                        token.set_embedding(self.name, subtoken_embeddings[token_idx])
                    else:
                        # otherwise, do a mean over all subwords in token
                        embeddings = subtoken_embeddings[
                            token_idx : token_idx
                            + feature.token_subtoken_count[token.idx]
                        ]
                        embeddings = [
                            embedding.unsqueeze(0) for embedding in embeddings
                        ]
                        try:
                            mean = torch.mean(torch.cat(embeddings, dim=0), dim=0)
                        except:
                            pdb.set_trace()
                        token.set_embedding(self.name, mean)

                    token_idx += feature.token_subtoken_count[token.idx] - 1
        if hasattr(sentences, 'features'):
            sentences = self.assign_batch_features(sentences)
        return sentences
    def set_batch_features(self,sentences):
        pass
    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        return (
            len(self.layer_indexes) * self.model.config.hidden_size
            if not self.use_scalar_mix
            else self.model.config.hidden_size
        )

