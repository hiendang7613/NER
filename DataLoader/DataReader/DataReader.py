import numpy as np
from bs4 import BeautifulSoup
import re

class DataReader(object):
    def __init__(self):
        pass

    def readFile(self, file_path):
        pass

    def ParseAllDocs(self):
        pass

    def getEntityType(self) -> []:
        pass

    def getTokenIds(self) -> []:
        pass

    def getLabels(self) -> np.ndarray:
        pass

    def preprocessing(self, text):
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return text



