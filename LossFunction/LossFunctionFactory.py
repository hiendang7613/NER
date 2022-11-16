from LossFunction.crf_loss import crf_log_likelihood

class LossFunctionFactory(object):
    def __init__(self):
        super(LossFunctionFactory, self).__init__()
        pass

    '''
    ## getHead parameters
    config = {
        'loss_func_type' : 
    }'''
    @staticmethod
    def getLossFunction(self, config):
        loss_func = None
        if config['loss_func_type'] == 'crf_log_likelihood':
            loss_func = crf_log_likelihood
        # elif backbone_type == 'backbone_1':
        #     output = BiLSTM(include_top=False)(input_tensor)
        # else:
        #     print('Head type not match!')
        return loss_func