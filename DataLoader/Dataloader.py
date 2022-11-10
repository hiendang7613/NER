from DataReader.DataReader import DataReader
from typing import List
import numpy as np

class Dataloader(object):
    def __init__(self, dataReaders: List[DataReader]):
        self.dataReaders = dataReaders
        self.dataset = None

        self.doc_token_ids = []
        self.doc_labels = []
        self.documents = []
        self.entity_type = []
        pass

    def readData(self):
        for reader in self.dataReaders:
            reader.ParseAllDocs()
            self.doc_token_ids.append(reader.getTokenIds())
            self.entity_type.append(reader.getEntityType())

        # merge array entity type
        self.entity_type = list(dict.fromkeys(self.entity_type))

        for reader in self.dataReaders:
            new_pos = [self.entity_type.index(e) for e in reader.getEntityType()]
            for doc_labels in reader.getLabels():
                new_doc_labels = np.empty((len(self.entity_type), doc_labels.shape[1]))
                for k in range(doc_labels.shape[0]):
                    new_doc_labels[new_pos[k]] = doc_labels[k]
                self.doc_labels.append(new_doc_labels)

    def getDataset(self):
        dataSet
        return self.dataset