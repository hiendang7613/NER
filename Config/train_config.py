from pathlib import Path
import tensorflow as tf

# data
projects_path = Path().parent.resolve()
data_path = projects_path / 'Dataset' / 'raw_tfrecords' / 'lfw.tfrecords'
text_max_len = 512


# train dataloader
epochs = 10
is_shuffle = False
buffer_size = 10240
shuffle_seed = 43
is_repeat = True
batch_size = 32
batch_drop_remainder=True
is_prefetch=True
is_prefetch_buffle=tf.data.experimental.AUTOTUNE

# preprocessing
is_preprocessing = True

# data augmentation
is_augmentation = False



# trainer
trainer_name = 'Trainer'
checkpoint_steps_num = 1000
training_dir = r'./Tensorboard'
export_dir = r'./Model'
monitor = 'categorical_accuracy'
mode = 'max'
epochs = 10
