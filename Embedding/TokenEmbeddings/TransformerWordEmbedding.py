from Embedding.TokenEmbeddings.TokenEmbedding import TokenEmbedding

class TransformerWordEmbeddings(TokenEmbedding):
    def __init__(
            self,
            model: str = "bert-base-uncased",
            layers: str = "-1,-2,-3,-4",
            pooling_operation: str = "first",
            batch_size: int = 1,
            use_scalar_mix: bool = False,
            fine_tune: bool = False,
            allow_long_sentences: bool = True,
            stride: int = -1,
            maximum_window: bool = False,
            document_extraction: bool = False,
            embedding_name: str = None,
            doc_batch_size: int = 32,
            maximum_subtoken_length: int = 999,
            v2_doc: bool = False,
            ext_doc: bool = False,
            sentence_feat: bool = False,
            **kwargs
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either take the first
        subtoken ('first'), the last subtoken ('last'), both first and last ('first_last') or a mean over all ('mean')
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        :param embedding_name: We recommend to set embedding_name if you use absolute path to the embedding file. If you do not set it in training, the order of embeddings is changed when you run the trained ACE model on other server.
        :param maximum_subtoken_length: The maximum length of subtokens for a token, if chunk the subtokens to the maximum length if it is longer than the maximum subtoken length.
        """
        super().__init__()

        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)
        self.model = AutoModel.from_pretrained(model, config=config, **kwargs)

        self.allow_long_sentences = allow_long_sentences
        if not hasattr(self.tokenizer, 'model_max_length'):
            self.tokenizer.model_max_length = 512
        if allow_long_sentences:
            self.max_subtokens_sequence_length = self.tokenizer.model_max_length
            self.stride = self.tokenizer.model_max_length // 2
            if stride != -1:
                if not maximum_window:
                    self.max_subtokens_sequence_length = stride * 2
                self.stride = stride
        else:
            self.max_subtokens_sequence_length = self.tokenizer.model_max_length
            self.stride = 0

        # model name
        # self.name = 'transformer-word-' + str(model)
        if embedding_name is None:
            self.name = str(model)
        else:
            self.name = embedding_name
            self.model_path = str(model)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(flair.device)

        # embedding parameters
        if layers == 'all':
            # send mini-token through to check how many layers the model has
            hidden_states = self.model(torch.tensor([1], device=flair.device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(",")]
        # self.mix = ScalarMix(mixture_size=len(self.layer_indexes), trainable=False)
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size
        self.sentence_feat = sentence_feat

        self.special_tokens = []
        # check if special tokens exist to circumvent error message
        try:
            if self.tokenizer._bos_token:
                self.special_tokens.append(self.tokenizer.bos_token)
            if self.tokenizer._cls_token:
                self.special_tokens.append(self.tokenizer.cls_token)
        except:
            pass
        self.document_extraction = document_extraction
        self.v2_doc = v2_doc
        self.ext_doc = ext_doc
        if self.v2_doc:
            self.name = self.name + '_v2doc'
        if self.ext_doc:
            self.name = self.name + '_extdoc'
        self.doc_batch_size = doc_batch_size
        # most models have an intial BOS token, except for XLNet, T5 and GPT2
        self.begin_offset = 1
        if type(self.tokenizer) == XLNetTokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == T5Tokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == GPT2Tokenizer:
            self.begin_offset = 0
        self.maximum_subtoken_length = maximum_subtoken_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences."""
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
        # split into micro batches of size self.batch_size before pushing through transformer
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]
        # if self.name == '/nas-alitranx/yongjiang.jy/wangxy/transformers/nl-bert_10epoch_0.5inter_500batch_0.00005lr_20lrrate_nl_monolingual_nocrf_fast_warmup_freezing_beta_weightdecay_finetune_saving_nodev_iwpt21_enhancedud14/bert-base-dutch-cased' or self.name == '/nas-alitranx/yongjiang.jy/wangxy/transformers/nl-xlmr-first_10epoch_0.5inter_1batch_4accumulate_0.000005lr_20lrrate_nl_monolingual_nocrf_fast_warmup_freezing_beta_weightdecay_finetune_saving_sentbatch_nodev_iwpt21_enhancedud16/robbert-v2-dutch-base':
        #     self.max_subtokens_sequence_length = 510
        #     self.stride = 255
        #     # pdb.set_trace()
        if hasattr(self, 'v2_doc') and self.v2_doc:
            model_max_length = self.tokenizer.model_max_length - 2
            if model_max_length > 510:
                model_max_length = 510
            self.add_document_embeddings_v2(sentences, max_sequence_length=model_max_length,
                                            batch_size=32 if not hasattr(self,
                                                                         'doc_batch_size') else self.doc_batch_size)
        elif self.ext_doc:
            orig_sentences = [sentence.orig_sent for idx, sentence in enumerate(sentences)]
            self._add_embeddings_to_sentences(orig_sentences)
            for sent_id, sentence in enumerate(sentences):
                orig_sentence = orig_sentences[sent_id]
                for token_id, token in enumerate(sentence):
                    token._embeddings[self.name] = orig_sentence[token_id]._embeddings[self.name]
            store_embeddings(orig_sentences, 'none')
        elif not hasattr(self, 'document_extraction') or not self.document_extraction:
            self._add_embeddings_to_sentences(sentences)
        else:
            # embed each micro-batch
            for batch in sentence_batches:
                self.add_document_embeddings(batch, stride=self.stride, batch_size=32 if not hasattr(self,
                                                                                                     'doc_batch_size') else self.doc_batch_size)
        if hasattr(sentences, 'features'):
            # store_embeddings(sentences, 'cpu')

            sentences = self.assign_batch_features(sentences)
        return sentences

    @staticmethod
    def _remove_special_markup(text: str):
        # remove special markup
        text = re.sub('^Ġ', '', text)  # RoBERTa models
        text = re.sub('^##', '', text)  # BERT models
        text = re.sub('^▁', '', text)  # XLNet models
        text = re.sub('</w>$', '', text)  # XLM models

        return text

    def _get_processed_token_text(self, token: Token) -> str:
        pieces = self.tokenizer.tokenize(token.text)
        token_text = ''
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

    def _add_embeddings_to_sentences(self, sentences: List[Sentence]):
        """Match subtokenization to Flair tokenization and extract embeddings from transformers for each token."""

        # keep a copy of sentences
        input_sentences = sentences
        # first, subtokenize each sentence and find out into how many subtokens each token was divided
        subtokenized_sentences = []
        subtokenized_sentences_token_lengths = []

        sentence_parts_lengths = []

        # TODO: keep for backwards compatibility, but remove in future
        # some pretrained models do not have this property, applying default settings now.
        # can be set manually after loading the model.
        if not hasattr(self, 'max_subtokens_sequence_length'):
            self.max_subtokens_sequence_length = None
            self.allow_long_sentences = False
            self.stride = 1

        non_empty_sentences = []
        empty_sentences = []
        batch_size = len(sentences)
        for sent_idx, sentence in enumerate(sentences):

            tokenized_string = sentence.to_tokenized_string()

            if '<EOS>' in tokenized_string:  # replace manually set <EOS> token to the EOS token of the tokenizer
                sent_tokens = copy.deepcopy(sentence.tokens)
                for token_id, token in enumerate(sent_tokens):
                    if token.text == '<EOS>':
                        if self.tokenizer._eos_token is not None:
                            if hasattr(self.tokenizer._eos_token, 'content'):
                                token.text = self.tokenizer._eos_token.content
                            else:
                                token.text = self.tokenizer._eos_token
                        elif self.tokenizer._sep_token is not None:
                            if hasattr(self.tokenizer._sep_token, 'content'):
                                token.text = self.tokenizer._sep_token.content
                            else:
                                token.text = self.tokenizer._sep_token

                if self.tokenizer._eos_token is not None:
                    if hasattr(self.tokenizer._eos_token, 'content'):
                        tokenized_string = re.sub('<EOS>', self.tokenizer._eos_token.content, tokenized_string)
                    else:
                        tokenized_string = re.sub('<EOS>', self.tokenizer._eos_token, tokenized_string)
                elif self.tokenizer._sep_token is not None:
                    if hasattr(self.tokenizer._sep_token, 'content'):
                        tokenized_string = re.sub('<EOS>', self.tokenizer._sep_token.content, tokenized_string)
                    else:
                        tokenized_string = re.sub('<EOS>', self.tokenizer._sep_token, tokenized_string)
                else:
                    pdb.set_trace()
            else:
                sent_tokens = sentence.tokens
            # method 1: subtokenize sentence
            # subtokenized_sentence = self.tokenizer.encode(tokenized_string, add_special_tokens=True)

            # method 2:
            # transformer specific tokenization
            subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)
            if len(subtokenized_sentence) == 0:
                empty_sentences.append(sentence)
                continue
            else:
                non_empty_sentences.append(sentence)
            # token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sentence, subtokenized_sentence)
            # pdb.set_trace()
            token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sent_tokens, subtokenized_sentence)
            token_subtoken_lengths = torch.LongTensor(token_subtoken_lengths)
            if (token_subtoken_lengths > self.maximum_subtoken_length).any():
                new_subtokenized_sentence = []
                current_idx = 0
                for subtoken_length in token_subtoken_lengths:
                    if subtoken_length > self.maximum_subtoken_length:
                        # pdb.set_trace()
                        new_subtokenized_sentence += subtokenized_sentence[
                                                     current_idx:current_idx + self.maximum_subtoken_length]
                        # current_idx += self.maximum_subtoken_length
                    else:
                        new_subtokenized_sentence += subtokenized_sentence[current_idx:current_idx + subtoken_length]
                    current_idx += subtoken_length
                token_subtoken_lengths[
                    torch.where(token_subtoken_lengths > self.maximum_subtoken_length)] = self.maximum_subtoken_length
                # pdb.set_trace()
                subtokenized_sentence = new_subtokenized_sentence

            subtokenized_sentences_token_lengths.append(token_subtoken_lengths)

            subtoken_ids_sentence = self.tokenizer.convert_tokens_to_ids(subtokenized_sentence)
            # if hasattr(self, 'output_num_feats') and self.output_num_feats:
            #     image_feat_idx = list(range(self.model.embeddings.word_embeddings.num_embeddings-self.output_num_feats*batch_size+sent_idx*self.output_num_feats,self.model.embeddings.word_embeddings.num_embeddings-self.output_num_feats*batch_size+(sent_idx+1)*self.output_num_feats))
            #     subtoken_ids_sentence += image_feat_idx
            nr_sentence_parts = 0
            if hasattr(self.tokenizer, 'encode_plus'):
                while subtoken_ids_sentence:

                    nr_sentence_parts += 1
                    # need to set the window size and stride freely
                    # encoded_inputs = self.tokenizer.encode_plus(subtoken_ids_sentence,
                    #                                             max_length=None,
                    #                                             stride=self.stride,
                    #                                             return_overflowing_tokens=self.allow_long_sentences,
                    #                                             truncation=False,
                    #                                             truncation_strategy = 'only_first',
                    #                                             )
                    encoded_inputs = self.tokenizer.encode_plus(subtoken_ids_sentence,
                                                                max_length=self.max_subtokens_sequence_length,
                                                                stride=self.stride,
                                                                return_overflowing_tokens=self.allow_long_sentences,
                                                                truncation=True,
                                                                )
                    # encoded_inputs = self.tokenizer.encode_plus(subtoken_ids_sentence,max_length=self.max_subtokens_sequence_length,stride=self.max_subtokens_sequence_length//2,return_overflowing_tokens=self.allow_long_sentences,truncation=True,)
                    subtoken_ids_split_sentence = encoded_inputs['input_ids']
                    subtokenized_sentences.append(torch.tensor(subtoken_ids_split_sentence, dtype=torch.long))

                    if 'overflowing_tokens' in encoded_inputs:
                        subtoken_ids_sentence = encoded_inputs['overflowing_tokens']
                    else:
                        subtoken_ids_sentence = None
            else:
                nr_sentence_parts += 1
                subtokenized_sentences.append(torch.tensor(subtoken_ids_sentence, dtype=torch.long))
            sentence_parts_lengths.append(nr_sentence_parts)
        # empty sentences get zero embeddings
        for sentence in empty_sentences:
            for token in sentence:
                token.set_embedding(self.name, torch.zeros(self.embedding_length))

        # only embed non-empty sentences and if there is at least one
        sentences = non_empty_sentences
        if len(sentences) == 0: return

        # find longest sentence in batch
        longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

        total_sentence_parts = sum(sentence_parts_lengths)
        # initialize batch tensors and mask
        input_ids = torch.zeros(
            [total_sentence_parts, longest_sequence_in_batch],
            dtype=torch.long,
            device=flair.device,
        )
        mask = torch.zeros(
            [total_sentence_parts, longest_sequence_in_batch],
            dtype=torch.long,
            device=flair.device,
        )
        for s_id, sentence in enumerate(subtokenized_sentences):
            sequence_length = len(sentence)
            input_ids[s_id][:sequence_length] = sentence
            mask[s_id][:sequence_length] = torch.ones(sequence_length)
        # put encoded batch through transformer model to get all hidden states of all encoder layers
        inputs_embeds = None
        if hasattr(input_sentences, 'img_features') and 'img_feats' in input_sentences.img_features:
            word_embeddings = self.model.embeddings.word_embeddings(input_ids)
            img_feats = input_sentences.img_features['img_feats'].to(flair.device)
            inputs_embeds = torch.cat([word_embeddings, img_feats], 1)
            image_mask = torch.ones([batch_size, img_feats.shape[1]]).type_as(mask)
            new_mask = torch.cat([mask, image_mask], -1)
            mask = new_mask
            # input_ids = None
        if 'xlnet' in self.name:
            hidden_states = self.model(input_ids, attention_mask=mask, inputs_embeds=inputs_embeds)[-1]
            if self.sentence_feat:
                assert 0, 'not implemented'
        else:
            sequence_output, pooled_output, hidden_states = self.model(input_ids, attention_mask=mask,
                                                                       inputs_embeds=inputs_embeds)
            if self.sentence_feat:
                self.pooled_output = pooled_output
            # hidden_states = self.model(input_ids, attention_mask=mask)[-1]

        # make the tuple a tensor; makes working with it easier.
        hidden_states = torch.stack(hidden_states)

        sentence_idx_offset = 0

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # iterate over all subtokenized sentences
            for sentence_idx, (sentence, subtoken_lengths, nr_sentence_parts) in enumerate(
                    zip(sentences, subtokenized_sentences_token_lengths, sentence_parts_lengths)):

                sentence_hidden_state = hidden_states[:, sentence_idx + sentence_idx_offset, ...]

                for i in range(1, nr_sentence_parts):
                    sentence_idx_offset += 1
                    remainder_sentence_hidden_state = hidden_states[:, sentence_idx + sentence_idx_offset, ...]
                    # remove stride_size//2 at end of sentence_hidden_state, and half at beginning of remainder,
                    # in order to get some context into the embeddings of these words.
                    # also don't include the embedding of the extra [CLS] and [SEP] tokens.
                    sentence_hidden_state = torch.cat((sentence_hidden_state[:, :-1 - self.stride // 2, :],
                                                       remainder_sentence_hidden_state[:, 1 + self.stride // 2:, :]), 1)
                subword_start_idx = self.begin_offset

                # for each token, get embedding
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):

                    # some tokens have no subtokens at all (if omitted by BERT tokenizer) so return zero vector
                    if number_of_subtokens == 0:
                        token.set_embedding(self.name, torch.zeros(self.embedding_length))
                        continue

                    subword_end_idx = subword_start_idx + number_of_subtokens

                    subtoken_embeddings: List[torch.FloatTensor] = []

                    # get states from all selected layers, aggregate with pooling operation
                    for layer in self.layer_indexes:
                        current_embeddings = sentence_hidden_state[layer][subword_start_idx:subword_end_idx]

                        if self.pooling_operation == "first":
                            final_embedding: torch.FloatTensor = current_embeddings[0]

                        if self.pooling_operation == "last":
                            final_embedding: torch.FloatTensor = current_embeddings[-1]

                        if self.pooling_operation == "first_last":
                            final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                        if self.pooling_operation == "mean":
                            all_embeddings: List[torch.FloatTensor] = [
                                embedding.unsqueeze(0) for embedding in current_embeddings
                            ]
                            final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                        subtoken_embeddings.append(final_embedding)

                    # use scalar mix of embeddings if so selected
                    if self.use_scalar_mix:
                        sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                        # sm_embeddings = self.mix(subtoken_embeddings)

                        subtoken_embeddings = [sm_embeddings]

                    # set the extracted embedding for the token
                    token.set_embedding(self.name, torch.cat(subtoken_embeddings))

                    subword_start_idx += number_of_subtokens

    def reconstruct_tokens_from_subtokens(self, sentence, subtokens):
        word_iterator = iter(sentence)
        token = next(word_iterator)
        token_text = self._get_processed_token_text(token)
        token_subtoken_lengths = []
        reconstructed_token = ''
        subtoken_count = 0
        # iterate over subtokens and reconstruct tokens
        for subtoken_id, subtoken in enumerate(subtokens):

            # remove special markup
            subtoken = self._remove_special_markup(subtoken)

            # TODO check if this is necessary is this method is called before prepare_for_model
            # check if reconstructed token is special begin token ([CLS] or similar)
            if subtoken in self.special_tokens and subtoken_id == 0:
                continue

            # some BERT tokenizers somehow omit words - in such cases skip to next token
            if subtoken_count == 0 and not token_text.startswith(subtoken.lower()):

                while True:
                    token_subtoken_lengths.append(0)
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                    if token_text.startswith(subtoken.lower()): break

            subtoken_count += 1

            # append subtoken to reconstruct token
            reconstructed_token = reconstructed_token + subtoken

            # check if reconstructed token is the same as current token
            if reconstructed_token.lower() == token_text:

                # if so, add subtoken count
                token_subtoken_lengths.append(subtoken_count)

                # reset subtoken count and reconstructed token
                reconstructed_token = ''
                subtoken_count = 0

                # break from loop if all tokens are accounted for
                if len(token_subtoken_lengths) < len(sentence):
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                else:
                    break

        # if tokens are unaccounted for
        while len(token_subtoken_lengths) < len(sentence) and len(token.text) == 1:
            token_subtoken_lengths.append(0)
            if len(token_subtoken_lengths) == len(sentence): break
            token = next(word_iterator)

        # check if all tokens were matched to subtokens
        if token != sentence[-1]:
            log.error(f"Tokenization MISMATCH in sentence '{sentence.to_tokenized_string()}'")
            log.error(f"Last matched: '{token}'")
            log.error(f"Last sentence: '{sentence[-1]}'")
            log.error(f"subtokenized: '{subtokens}'")
        return token_subtoken_lengths

    def train(self, mode=True):
        # if fine-tuning is not enabled (i.e. a "feature-based approach" used), this
        # module should never be in training mode
        if not self.fine_tune:
            pass
        else:
            super().train(mode)

    def convert_example_to_features(self, example, window_start, window_end, tokens_ids_to_extract, tokenizer,
                                    seq_length):
        # there is no [SEP] and [CLS]
        window_tokens = example.tokens[window_start:window_end]

        tokens = []
        input_type_ids = []
        for token in window_tokens:
            tokens.append(token)
            input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        extract_indices = [-1] * seq_length
        for i in tokens_ids_to_extract:
            assert i - window_start >= 0
            extract_indices[i - window_start] = i

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return dict(unique_ids=example.document_index,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    extract_indices=extract_indices)

    def add_document_embeddings(self, sentences: List[Sentence], window_size=511, stride=1, batch_size=32):
        # Sentences: a group of sentences that forms a document

        # first, subtokenize each sentence and find out into how many subtokens each token was divided

        # subtokenized_sentences = []
        # subtokenized_sentences_token_lengths = []

        # sentence_parts_lengths = []

        # # TODO: keep for backwards compatibility, but remove in future
        # # some pretrained models do not have this property, applying default settings now.
        # # can be set manually after loading the model.
        # if not hasattr(self, 'max_subtokens_sequence_length'):
        #     self.max_subtokens_sequence_length = None
        #     self.allow_long_sentences = False
        #     self.stride = 0
        non_empty_sentences = []
        empty_sentences = []
        doc_token_subtoken_lengths = []
        doc_subtoken_ids_sentence = []
        for sentence in sentences:
            tokenized_string = sentence.to_tokenized_string()

            sent_tokens = copy.deepcopy(sentence.tokens)
            if '<EOS>' in tokenized_string:  # replace manually set <EOS> token to the EOS token of the tokenizer
                for token_id, token in enumerate(sent_tokens):
                    if token.text == '<EOS>':
                        if self.tokenizer._eos_token is not None:
                            token.text = self.tokenizer._eos_token
                        elif self.tokenizer._sep_token is not None:
                            token.text = self.tokenizer._sep_token

                if self.tokenizer._eos_token is not None:
                    tokenized_string = re.sub('<EOS>', self.tokenizer._eos_token, tokenized_string)
                elif self.tokenizer._sep_token is not None:
                    tokenized_string = re.sub('<EOS>', self.tokenizer._sep_token, tokenized_string)
                else:
                    pdb.set_trace()
            # method 1: subtokenize sentence
            # subtokenized_sentence = self.tokenizer.encode(tokenized_string, add_special_tokens=True)

            # method 2:
            # transformer specific tokenization
            subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)
            if len(subtokenized_sentence) == 0:
                empty_sentences.append(sentence)
                continue
            else:
                non_empty_sentences.append(sentence)

            # token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sentence, subtokenized_sentence)
            token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sent_tokens, subtokenized_sentence)
            doc_token_subtoken_lengths.append(token_subtoken_lengths)

            subtoken_ids_sentence = self.tokenizer.convert_tokens_to_ids(subtokenized_sentence)
            doc_subtoken_ids_sentence.append(subtoken_ids_sentence)
        doc_subtokens = []
        for subtokens in doc_subtoken_ids_sentence:
            doc_subtokens += subtokens

        doc_sentence = []
        for subtokens in doc_sentence:
            doc_sentence += subtokens
        doc_input_ids = []
        doc_input_masks = []
        doc_hidden_states = []
        doc_token_ids_to_extract = []
        doc_extract_indices = []
        for i in range(0, len(doc_subtokens), stride):
            # if i == len(example.tokens)-1:
            #     pdb.set_trace()
            if i % batch_size == 0:
                if i != 0:
                    doc_input_ids.append(batch_input_ids)
                    doc_input_masks.append(batch_mask)
                batch_input_ids = torch.zeros(
                    [batch_size, window_size],
                    dtype=torch.long,
                    device=flair.device,
                )
                batch_mask = torch.zeros(
                    [batch_size, window_size],
                    dtype=torch.long,
                    device=flair.device,
                )
            window_center = i + window_size // 2
            token_ids_to_extract = []
            extract_start = int(np.clip(window_center - stride // 2, 0, len(doc_subtokens)))
            extract_end = int(np.clip(window_center + stride // 2 + 1, extract_start, len(doc_subtokens)))

            if i == 0:
                token_ids_to_extract.extend(range(extract_start))
            # position in the doc_subtokens
            token_ids_to_extract.extend(range(extract_start, extract_end))
            if token_ids_to_extract == []:
                break
            if i + stride >= len(doc_subtokens):
                token_ids_to_extract.extend(range(extract_end, len(doc_subtokens)))
            doc_token_ids_to_extract.append(token_ids_to_extract.copy())
            window_start = i
            window_end = min(i + window_size, len(doc_subtokens))
            input_ids = torch.Tensor(doc_subtokens[window_start:window_end]).type_as(batch_input_ids)
            # pdb.set_trace()
            mask = torch.ones_like(input_ids).type_as(batch_mask)
            batch_input_ids[i % batch_size, :len(input_ids)] = input_ids
            batch_mask[i % batch_size, :len(mask)] = mask
            # position in extracted features
            extract_indices = [idx - window_start for idx in token_ids_to_extract]
            for idx in extract_indices:
                assert idx >= 0
            # for idx in tokens_ids_to_extract:
            #     assert idx - window_start >= 0
            #     extract_indices.append(idx - window_start)
            doc_extract_indices.append(extract_indices.copy())
            # input_ids, mask, self.convert_example_to_features(doc_subtokens,i,min(i + window_size, len(doc_subtokens)),token_ids_to_extract,self.tokenizer,window_size)

            # # find longest sentence in batch
            # longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

            # total_sentence_parts = sum(sentence_parts_lengths)
            # # initialize batch tensors and mask

            # for s_id, sentence in enumerate(subtokenized_sentences):
            #     sequence_length = len(sentence)
            #     input_ids[s_id][:sequence_length] = sentence
            #     mask[s_id][:sequence_length] = torch.ones(sequence_length)
            # # put encoded batch through transformer model to get all hidden states of all encoder layers
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()
        # sublens=[sum(x) for x in doc_token_subtoken_lengths]
        with gradient_context:

            # pdb.set_trace()
            # assert sum([len(x) for x in doc_extract_indices]) == len(doc_subtokens)
            if batch_input_ids.sum() != 0:
                doc_input_ids.append(batch_input_ids)
                doc_input_masks.append(batch_mask)
            doc_hidden_states = torch.zeros([len(doc_subtokens), self.embedding_length])
            for i in range(len(doc_input_ids)):
                hidden_states = torch.stack(self.model(doc_input_ids[i], attention_mask=doc_input_masks[i])[-1])[
                    self.layer_indexes]
                hidden_states = hidden_states.permute([1, 2, 3, 0])
                # reshape to batch x subtokens x hidden_size*layers
                # pdb.set_trace()
                # hidden_states = hidden_states.reshape(hidden_states.shape[0],hidden_states.shape[1],-1)
                hidden_states = [hidden_states[:, :, :, x] for x in range(len(self.layer_indexes))]
                hidden_states = torch.cat(hidden_states, -1)
                hidden_states = hidden_states.cpu()
                for h_idx, hidden_state in enumerate(hidden_states):
                    if i * batch_size + h_idx >= len(doc_extract_indices):
                        break
                    try:
                        extract_indices = doc_extract_indices[i * batch_size + h_idx]
                        token_ids_to_extract = doc_token_ids_to_extract[i * batch_size + h_idx]
                        # assert len(extract_indices)==len(token_ids_to_extract)
                        doc_hidden_states[torch.Tensor(token_ids_to_extract).long()] = hidden_state[
                            torch.Tensor(extract_indices).long()]
                    except:
                        pdb.set_trace()
                # doc_hidden_states.append()
            # make the tuple a tensor; makes working with it easier.
            # iterate over all subtokenized sentences
            sentence_idx_offset = 0
            for sentence_idx, (sentence, subtoken_lengths) in enumerate(zip(sentences, doc_token_subtoken_lengths)):

                sentence_hidden_state = doc_hidden_states[
                                        sentence_idx_offset:sentence_idx_offset + sum(subtoken_lengths)]

                subword_start_idx = 0
                # for each token, get embedding
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):

                    # some tokens have no subtokens at all (if omitted by BERT tokenizer) so return zero vector
                    if number_of_subtokens == 0:
                        token.set_embedding(self.name, torch.zeros(self.embedding_length))
                        continue

                    subword_end_idx = subword_start_idx + number_of_subtokens

                    subtoken_embeddings: List[torch.FloatTensor] = []

                    current_embeddings = sentence_hidden_state[subword_start_idx:subword_end_idx]

                    if self.pooling_operation == "first":
                        final_embedding: torch.FloatTensor = current_embeddings[0]

                    if self.pooling_operation == "last":
                        final_embedding: torch.FloatTensor = current_embeddings[-1]

                    if self.pooling_operation == "first_last":
                        final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                    if self.pooling_operation == "mean":
                        all_embeddings: List[torch.FloatTensor] = [
                            embedding.unsqueeze(0) for embedding in current_embeddings
                        ]
                        final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                    # set the extracted embedding for the token
                    token.set_embedding(self.name, final_embedding)

                    subword_start_idx += number_of_subtokens
                sentence_idx_offset += subword_start_idx
        return sentences

    def add_document_embeddings_v2(self, sentences: List[Sentence], max_sequence_length=510, batch_size=32):
        # Sentences: a group of sentences that forms a document

        # first, subtokenize each sentence and find out into how many subtokens each token was divided

        # subtokenized_sentences = []
        # subtokenized_sentences_token_lengths = []

        # sentence_parts_lengths = []

        # # TODO: keep for backwards compatibility, but remove in future
        # # some pretrained models do not have this property, applying default settings now.
        # # can be set manually after loading the model.
        # if not hasattr(self, 'max_subtokens_sequence_length'):
        #     self.max_subtokens_sequence_length = None
        #     self.allow_long_sentences = False
        #     self.stride = 0
        non_empty_sentences = []
        empty_sentences = []
        doc_token_subtoken_lengths = []

        batch_doc_subtokens = []
        batch_pos = []
        # pdb.set_trace()
        for sentence in sentences:
            doc_subtokens = []

            if not hasattr(sentence, 'batch_pos'):
                sentence.batch_pos = {}
                sentence.target_tokens = {}
                sentence.token_subtoken_lengths = {}
            # pdb.set_trace()
            if self.name in sentence.batch_pos:
                start_pos, end_pos = sentence.batch_pos[self.name]
                target_tokens = sentence.target_tokens[self.name]
                doc_token_subtoken_lengths.append(sentence.token_subtoken_lengths[self.name])
            else:
                for doc_pos, doc_sent in enumerate(sentence.doc):
                    if doc_pos == sentence.doc_pos:
                        doc_sent_start = len(doc_subtokens)
                    doc_sent.doc_sent_start = len(doc_subtokens)
                    tokenized_string = doc_sent.to_tokenized_string()

                    # method 1: subtokenize sentence
                    # subtokenized_sentence = self.tokenizer.encode(tokenized_string, add_special_tokens=True)

                    # method 2:
                    # transformer specific tokenization

                    if not hasattr(doc_sent, 'subtokenized_sentence'):
                        doc_sent.subtokenized_sentence = {}
                    if self.name in doc_sent.subtokenized_sentence:
                        subtokenized_sentence = doc_sent.subtokenized_sentence[self.name]
                    else:
                        subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)
                        doc_sent.subtokenized_sentence[self.name] = subtokenized_sentence

                    if len(subtokenized_sentence) == 0:
                        empty_sentences.append(doc_sent)
                        continue
                    else:
                        non_empty_sentences.append(doc_sent)
                    if not hasattr(doc_sent, 'token_subtoken_lengths'):
                        doc_sent.token_subtoken_lengths = {}
                    if self.name in doc_sent.token_subtoken_lengths:
                        token_subtoken_lengths = doc_sent.token_subtoken_lengths[self.name]
                    else:
                        token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(doc_sent, subtokenized_sentence)
                        doc_sent.token_subtoken_lengths[self.name] = token_subtoken_lengths

                    # token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sent_tokens, subtokenized_sentence)
                    if doc_pos == sentence.doc_pos:
                        # sentence.token_subtoken_lengths[self.name] = token_subtoken_lengths
                        doc_token_subtoken_lengths.append(token_subtoken_lengths)

                    if not hasattr(doc_sent, 'subtoken_ids_sentence'):
                        doc_sent.subtoken_ids_sentence = {}
                    if self.name in doc_sent.subtoken_ids_sentence:
                        subtoken_ids_sentence = doc_sent.subtoken_ids_sentence[self.name]
                    else:
                        subtoken_ids_sentence = self.tokenizer.convert_tokens_to_ids(subtokenized_sentence)
                        doc_sent.subtoken_ids_sentence[self.name] = subtoken_ids_sentence

                    doc_subtokens += subtoken_ids_sentence
                    if doc_pos == sentence.doc_pos:
                        doc_sent_end = len(doc_subtokens)
                    doc_sent.doc_sent_end = len(doc_subtokens)

                left_length = doc_sent_start
                right_length = len(doc_subtokens) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_sequence_length - sentence_length) / 2)

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length,
                                               max_sequence_length - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_sequence_length - right_context_length - sentence_length)

                doc_offset = doc_sent_start - left_context_length
                target_tokens = doc_subtokens[doc_offset: doc_sent_end + right_context_length]
                target_tokens = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + target_tokens + [
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
                start_pos = doc_sent_start - doc_offset + 1
                end_pos = doc_sent_end - doc_offset + 1
                try:
                    assert start_pos >= 0
                    assert end_pos >= 0
                except:
                    print(sentences)

                sentence.batch_pos[self.name] = start_pos, end_pos
                sentence.target_tokens[self.name] = target_tokens

                # post-process for all sentences in the doc
                for doc_pos, doc_sent in enumerate(sentence.doc):
                    if not hasattr(doc_sent, 'batch_pos'):
                        doc_sent.batch_pos = {}
                        doc_sent.target_tokens = {}
                    if self.name in doc_sent.batch_pos:
                        continue
                    left_length = doc_sent.doc_sent_start
                    right_length = len(doc_subtokens) - doc_sent.doc_sent_end
                    sentence_length = doc_sent.doc_sent_end - doc_sent.doc_sent_start
                    half_context_length = max(int((max_sequence_length - sentence_length) / 2), 0)

                    if left_length < right_length:
                        left_context_length = min(left_length, half_context_length)
                        right_context_length = min(right_length,
                                                   max_sequence_length - left_context_length - sentence_length)
                    else:
                        right_context_length = min(right_length, half_context_length)
                        left_context_length = min(left_length,
                                                  max_sequence_length - right_context_length - sentence_length)
                    if left_context_length < 0:
                        left_context_length = 0
                    if right_context_length < 0:
                        right_context_length = 0

                    doc_offset = doc_sent.doc_sent_start - left_context_length
                    target_tokens = doc_subtokens[doc_offset: doc_sent.doc_sent_end + right_context_length]
                    target_tokens = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)] + target_tokens + [
                        self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
                    start_pos = doc_sent.doc_sent_start - doc_offset + 1
                    end_pos = doc_sent.doc_sent_end - doc_offset + 1
                    try:
                        assert start_pos >= 0
                        assert end_pos >= 0
                    except:
                        print(sentences)

                    doc_sent.batch_pos[self.name] = start_pos, end_pos
                    doc_sent.target_tokens[self.name] = target_tokens

                start_pos, end_pos = sentence.batch_pos[self.name]
                target_tokens = sentence.target_tokens[self.name]
                # doc_token_subtoken_lengths.append(sentence.token_subtoken_lengths[self.name])

            batch_doc_subtokens.append(target_tokens)
            batch_pos.append((start_pos, end_pos))

        input_lengths = [len(x) for x in batch_doc_subtokens]
        max_input_length = max(input_lengths)
        doc_input_ids = torch.zeros([len(sentences), max_input_length]).to(flair.device).long()
        doc_input_masks = torch.zeros([len(sentences), max_input_length]).to(flair.device).long()
        for i in range(len(sentences)):
            doc_input_ids[i, :input_lengths[i]] = torch.Tensor(batch_doc_subtokens[i]).type_as(doc_input_ids)
            doc_input_masks[i, :input_lengths[i]] = 1
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()
        # sublens=[sum(x) for x in doc_token_subtoken_lengths]
        # pdb.set_trace()
        with gradient_context:
            hidden_states = torch.stack(self.model(doc_input_ids, attention_mask=doc_input_masks)[-1])[
                self.layer_indexes]
            hidden_states = hidden_states.permute([1, 2, 3, 0])
            hidden_states = [hidden_states[:, :, :, x] for x in range(len(self.layer_indexes))]
            hidden_states = torch.cat(hidden_states, -1)
            # make the tuple a tensor; makes working with it easier.
            # iterate over all subtokenized sentences
            sentence_idx_offset = 0
            for sentence_idx, (sentence, subtoken_lengths) in enumerate(zip(sentences, doc_token_subtoken_lengths)):
                start_pos, end_pos = batch_pos[sentence_idx]
                sentence_hidden_state = hidden_states[sentence_idx, start_pos:end_pos]
                assert end_pos - start_pos == sum(doc_token_subtoken_lengths[sentence_idx])
                # sentence_hidden_state = doc_hidden_states[sentence_idx_offset:sentence_idx_offset+sum(subtoken_lengths)]

                subword_start_idx = 0
                # for each token, get embedding
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):

                    # some tokens have no subtokens at all (if omitted by BERT tokenizer) so return zero vector
                    if number_of_subtokens == 0:
                        token.set_embedding(self.name, torch.zeros(self.embedding_length))
                        continue

                    subword_end_idx = subword_start_idx + number_of_subtokens

                    subtoken_embeddings: List[torch.FloatTensor] = []

                    current_embeddings = sentence_hidden_state[subword_start_idx:subword_end_idx]
                    if self.pooling_operation == "first":
                        try:
                            final_embedding: torch.FloatTensor = current_embeddings[0]
                        except:
                            pdb.set_trace()

                    if self.pooling_operation == "last":
                        final_embedding: torch.FloatTensor = current_embeddings[-1]

                    if self.pooling_operation == "first_last":
                        final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                    if self.pooling_operation == "mean":
                        all_embeddings: List[torch.FloatTensor] = [
                            embedding.unsqueeze(0) for embedding in current_embeddings
                        ]
                        final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                    # set the extracted embedding for the token
                    token.set_embedding(self.name, final_embedding)

                    subword_start_idx += number_of_subtokens
                sentence_idx_offset += subword_start_idx
        return sentences

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

        if not self.use_scalar_mix:
            length = len(self.layer_indexes) * self.model.config.hidden_size
        else:
            length = self.model.config.hidden_size

        if self.pooling_operation == 'first_last': length *= 2

        return length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tokenizer"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # reload tokenizer to get around serialization issues
        model_name = self.name.split('transformer-word-')[-1]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            pass

