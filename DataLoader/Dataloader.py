from DataReader.DataReader import DataReader
from typing import List
import numpy as np
import tensorflow as tf

class Dataloader(object):
    def __init__(self, dataReaders: List[DataReader]):
        self.dataReaders = dataReaders
        self.dataset = None

        self.doc_token_ids = []
        self.doc_labels = []
        self.entity_type = []
        pass

    def getTokenIds(self):
        return self.doc_token_ids

    def getLabels(self):
        return self.doc_labels

    def getEntityType(self):
        return self.entity_type

    def getDataset(self):
        return self.dataset

    def getNumRecords(self):
        return len(self.doc_token_ids)

    def load(self):
        # read data
        for reader in self.dataReaders:
            reader.ParseAllDocs()
            self.doc_token_ids.extend(reader.getTokenIds())
            self.entity_type.append(reader.getEntityType())

        # merge array entity type
        self.entity_type = list(dict.fromkeys(self.entity_type))
        # merge data of multiple reader data
        for reader in self.dataReaders:
            new_pos = [self.entity_type.index(e) for e in reader.getEntityType()]
            for doc_labels in reader.getLabels():
                new_doc_labels = np.empty((len(self.entity_type), doc_labels.shape[1]))
                for k in range(doc_labels.shape[0]):
                    new_doc_labels[new_pos[k]] = doc_labels[k]
                self.doc_labels.append(new_doc_labels)
        
        self.dataset = tf.data.Dataset.from_tensor_slices((self.doc_token_ids, self.doc_labels))
        pass

    def addTFRecord(self, tfrecord_path):
        tfdataset = tf.data.TFRecordDataset(tfrecord_path)
        self.dataset = self.dataset.concatenate(tfdataset)
        pass


