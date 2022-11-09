import tensorflow as tf
from model import ModelFactory
from trainer import Trainer
from LossFunction.losses import CosfaceLoss
from Config import train_config, mlflow_config
import mlflow_log as mll


def create_teachers(self):
    teacher_list = []
    for corpus in self.corpus_list:
        config = Params.from_file(self.config[self.target][corpus]['train_config'])
        teacher_model = self.create_model(config, pretrained=True)
        teacher_model.targets = set([corpus])
        teacher_list.append(teacher_model)
    return teacher_list
def create_student():
    embeddings, word_map, char_map, lemma_map, postag_map = self.create_embeddings(config['embeddings'])
    tagger = getattr(models, classname)(**kwargs, config=config)
    tagger.word_map = word_map
    tagger.char_map = char_map
    tagger.lemma_map = lemma_map
    tagger.postag_map = postag_map
    tagger.use_bert = True
    tagger.use_crf = True

    return tagger

def trainning():
    student = create_student()
    teachers = create_teachers()

    trainer_func = trainer_func(student, teachers, corpus, config=config.config, professors=professors,
                                **config.config[trainer_name])

    for embedding in student.embeddings.embeddings:
        # manually fix the bug for the tokenizer becoming None
        if hasattr(embedding, 'tokenizer') and embedding.tokenizer is None:
            from transformers import AutoTokenizer
            name = embedding.name+
            embedding.tokenizer = AutoTokenizer.from_pretrained(name, use_fast = False)
    return x

def train(mlrun):

    # get hyper-parameter
    num_classes = train_config.num_classes
    embedding_size = train_config.embedding_size
    input_shape = train_config.input_shape


    # dataloader


    modelConfig = {
        'backbone_type': 'backbone_type',
        'head_type': 'head_type',
        # **'input_shape', 'embedding_size', 'num_classes'
    }
    model = ModelFactory.getModel(modelConfig)
    model.build(input_shape=input_shape)
    model.summary()

    # init loss function
    loss_fn = CosfaceLoss(margin=0.5, scale=64, n_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001)


    trainer = Trainer(train_dataloader=dataloader_train,
                        validation_dataloader=None,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        model=model)

    trainer.restore(weights_only=False, from_scout=True)
    trainer.train()
    trainer.export(model=model.backbone)
    # supervisor.mlflow_artifact(model=model.backbone,
    #                            tensorboard_dir=training_dir,
    #                            export_dir=export_dir)

    # evaluate ...
    # eval_class = EvaluteObjects(tfrecord_file=tfrecord_file_eval,
    #                             file_pairs=file_pair_eval,
    #                             batch_size=batch_size)
    # metrics = eval_class.activate(model=model.backbone, embedding_size=embedding_size)
    # eval_class.mlflow_logs(dict_metrics=metrics)




if __name__ == '__main__':
    mlrun = mll.mllog_run()
    train(mlrun)