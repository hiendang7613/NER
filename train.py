
import mlflow
import tensorflow as tf
import argparse
from model import ModelFactory
from Network.head.archead import ArcHead
from Tensorflow.TFRecord.tfrecord import TFRecordData
from training_supervisor import TrainingSupervisor
from LossFunction.losses import CosfaceLoss
from evalute import EvaluteObjects
from config import train_config, mlflow_config
from utlis.utlis import set_env_vars
import mlflow_log as mll

def train(mlrun):

    # get hyper-parameter
    tfrecord_file = train_config.tfrecord_file
    tfrecord_file_eval = train_config.tfrecord_file_eval
    file_pair_eval = train_config.file_pair_eval
    num_classes = train_config.num_classes
    num_images = train_config.num_images
    embedding_size = train_config.embedding_size
    batch_size = train_config.batch_size
    epochs = train_config.epochs
    input_shape = train_config.input_shape
    training_dir = train_config.training_dir
    export_dir = train_config.export_dir

    backbone_type = ''
    head_type = ''
    model = ModelFactory.getModel(backbone_type, head_type)


    archead = ArcHead(num_classes=num_classes)
    model = MyModel(type_backbone=type_backbone,
                    input_shape=input_shape,
                    embedding_size=embedding_size,
                    header=archead)
    model.build(input_shape=(None, input_shape, input_shape, 3))
    optimizer = tf.keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001)
    model.summary()

    # init loss function
    loss_fn = CosfaceLoss(margin=0.5, scale=64, n_classes=num_classes)

    # dataloader
    dataloader_train = TFRecordData.load(record_name=tfrecord_file,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         is_repeat=False,
                                         binary_img=True,
                                         is_crop=True,
                                         reprocess=False,
                                         num_classes=num_classes,
                                         buffer_size=2048)

    supervisor = TrainingSupervisor(train_dataloader=dataloader_train,
                                    validation_dataloader=None,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    model=model,
                                    save_freq=1000,
                                    monitor='categorical_accuracy',
                                    mode='max',
                                    training_dir=training_dir,
                                    name='Trainer_Supervisor')

    supervisor.restore(weights_only=False, from_scout=True)
    supervisor.train(epochs=epochs, steps_per_epoch=num_images // batch_size)
    supervisor.export(model=model.backbone, export_dir=export_dir)
    supervisor.mlflow_artifact(model=model.backbone,
                               tensorboard_dir=training_dir,
                               export_dir=export_dir)

    # evaluate ...
    eval_class = EvaluteObjects(tfrecord_file=tfrecord_file_eval,
                                file_pairs=file_pair_eval,
                                batch_size=batch_size)
    metrics = eval_class.activate(model=model.backbone, embedding_size=embedding_size)
    eval_class.mlflow_logs(dict_metrics=metrics)




if __name__ == '__main__':
    set_env_vars()
    mlrun = mll.mllog_run()
    mlflow_run()