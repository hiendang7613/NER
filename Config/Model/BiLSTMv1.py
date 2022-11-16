
## Embedding
embedding_type = 'phobert'
# embedding_size = 768 # phobert default

## Backbone
backbone_type = 'BiLSTM'
input_shape = 512
output_embedding_size = 256

## Head
head_type = 'crf'
num_classes = 3

## Loss function
loss_func_type = 'crf_log_likelihood'

