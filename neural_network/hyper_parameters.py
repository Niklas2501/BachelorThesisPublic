import json


class HyperParams:

    def __init__(self):
        pass

    def get_uniwarp_config(self):
        config = {'optimizer:num_epochs': 2000000,
                  'model:num_batch_pairs': 64,
                  'uniwarp:length': 1024,
                  'uniwarp:rnn_encoder_layers': [192, 96, 48],  # [128, 64, 48],  # [256, 128, 64],
                  'uniwarp:warp_nn_layers': [64, 16, 1],  # [48, 32, 1],  # [48, 16, 1],  #
                  'uniwarp:eta': 0.0001,
                  'uniwarp:max_grad_norm': 10.0,
                  'uniwarp:lambda': 0.0,
                  'uniwarp:cnn_encoder_layers': [1024, 256, 64],
                  'uniwarp:cnn_kernel_lengths': [5, 5, 3],
                  'uniwarp:cnn_strides': [2, 1, 1],
                  'uniwarp:dropout_rate': 0.05,
                  'uniwarp:enable_batch_normalization': True,
                  'dataset:num_channels': 1}

        return config

    @staticmethod
    def restore(file_path):
        return json.loads(file_path)
