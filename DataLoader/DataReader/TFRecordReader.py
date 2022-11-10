import numpy as np

class TFRecordReader(object):
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





def active(path_output_record):
    logging.info(f"Writing TFRecord file ...")
    with tf.io.TFRecordWriter(path=path_output_record) as writer:
        for img_path, real_name, filename, id_name in tqdm(TFRecordData.samples):
            sample = TFSample.create(np_image=open(img_path, 'rb').read(),
                                     id_name=int(id_name),
                                     filename=str.encode(filename),
                                     img_path=str.encode(img_path),
                                     real_name=str.encode(real_name))
            writer.write(record=sample.SerializeToString())
    logging.info(f"Writing TFRecord file done.")


@staticmethod
def create(path_dataset):
    """
    Create TFRecord file
    :return:
    """
    if not os.path.isdir(path_dataset):
        logging.error("Please check path dataset is not a folder.")
    else:
        logging.info(f"Creating TFRecord file from: {path_dataset} ... ")

    logging.info(f"Reading data list ...")
    for index_name, real_name in tqdm(enumerate(os.listdir(path_dataset))):
        list_id_filename = glob.glob(os.path.join(path_dataset, real_name, "*.jpg"))
        for img_path in list_id_filename:
            filename = os.path.join(real_name, os.path.basename(img_path))
            TFRecordData.samples.append((img_path, real_name, filename, TFRecordData.num_class_count))

        TFRecordData.num_class_count += 1
    random.shuffle(TFRecordData.samples)

    logging.info(f"Reading data list done.")

    logging.info(f"Creating TFRecord file done.")