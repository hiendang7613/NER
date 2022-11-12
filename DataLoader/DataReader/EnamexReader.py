import py_vncorenlp
import numpy as np

class EnamexReader(DataReader):
    def __init__(self, word_tokenizer, subword_tokenizer, cls_token='<s>', sep_token='</s>'):
        super().__init__()
        self.doc_token_ids = []
        self.doc_labels = []
        self.documents = []
        self.entity_type = []

        self.word_tokenizer = word_tokenizer
        self.subword_tokenizer = subword_tokenizer
        self.cls_token = cls_token
        self.sep_token = sep_token
        pass

    def readFile(self, file_path):
        with open(file_path, 'r') as f:
            document = f.read()
            self.append(document)
        pass

    def getTokenIds(self):
        return self.doc_token_ids

    def getLabels(self):
        return self.doc_labels

    def getEntityType(self):
        return self.entity_type

    def ParseAllDocs(self):
        for document in self.documents:
            onedoc_token_ids = []
            labels = []
            sentences = self.word_tokenizer(document)  # eg. Tôi là sinh_viên trường đại_học KHTN.

            # Add CLS token
            sentences[0] = self.cls_token + sentences[0]

            # Loop sentences in a document
            for sentence in sentences:

                # Add SEP token
                sentence = sentence + ' ' + self.sep_token

                sentence = sentence.replace('< ENAMEX TYPE= " ', '<ENAMEX_TYPE="') \
                    .replace(' " >', '">') \
                    .replace('< / ENAMEX >', '</ENAMEX>')
                # tokenize by white space to generate multiple label
                tokens = np.array(self.subword_tokenizer.tokenize(sentence))
                # [PERSON, LOCATION, ORGANIZATION]
                sen_labels = np.full((len(self.entity_type), len(tokens)), False)
                tags = np.full((len(tokens)), False)
                tag_level = []

                # Loop tokens in a sentence
                for index in range(len(tokens)):

                    if tags[index - 1] == False and tokens[index].startswith('<ENAMEX_TYPE'):
                        tags[index] = True
                    elif tags[index - 1] == True:
                        sen_labels[:, index] = sen_labels[:, index - 1]

                        if tokens[index - 1].startswith('<ENAMEX_TYPE'):
                            sen_labels[:, index - 1] = False
                            enamex_type = tokens[index - 1][14:-2]
                            if enamex_type not in self.entity_type:
                                self.entity_type.append(enamex_type)
                            type_id = self.entity_type.index(enamex_type)
                            sen_labels[type_id, index] = True
                            tag_level.append(type_id)
                        elif tokens[index - 1] == '</ENAMEX>':
                            sen_labels[:, index - 1] = False
                            highest_tag = tag_level.pop()
                            type_id = self.entity_type.index(highest_tag)
                            sen_labels[type_id, index] = False

                        tags[index] = len(tag_level) > 0
                    pass

                # Remove ENAMEX tags
                all_labels = np.bitwise_or.reduce(sen_labels, 0)
                tag_mask = np.logical_xor(tags, all_labels)

                # Replace dot sign by [SEP]
                if tokens[-2] == '.':
                    tag_mask[-2] = True

                # Masking
                sen_labels = sen_labels[:, ~tag_mask]
                tokens = tokens[~tag_mask]

                sent_tokenized_ids = self.subword_tokenizer.encode(tokens.tolist(), add_special_tokens=False)

                onedoc_token_ids.extend(sent_tokenized_ids)
                labels.append(sen_labels)
                pass

                labels = np.concatenate([np.resize(sen_labels,(len(self.entity_type),sen_labels.shape[1])) for sen_labels in labels], axis=1)
                    
            self.doc_token_ids.append(onedoc_token_ids)
            self.doc_labels.append(labels)
        pass
    pass
