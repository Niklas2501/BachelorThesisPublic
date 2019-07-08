import os
import sys
import tensorflow as tf

from configuration.Configuration import Configuration
from neural_network import rnn_models, cnn_models
from neural_network.dataset import Dataset
from neural_network.hyper_parameters import HyperParams
from neural_network.optimizer import Optimizer


def main():
    # suppress debugging messages of tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append("warp/")

    # get the configurations, fixed in the class hyper_parameters
    hp = HyperParams()
    hyper = hp.get_uniwarp_config()

    config = Configuration()

    # Import the training data from the folder into a dataset
    dataset_folder = config.preprocessed_folder
    dataset = Dataset()
    dataset.load_multivariate(dataset_folder)

    # change the values in the config depending on the loaded data
    hyper['uniwarp:length'] = dataset.series_length
    hyper['dataset:num_channels'] = dataset.num_channels

    # create the model
    model = None

    # create a neural network model depending on the given argument
    if config.architecture_type == 'SiameseRNN':
        model = rnn_models.SiameseRNN(hyper=hyper)
    elif config.architecture_type == 'WarpedSiameseRNN':
        model = rnn_models.WarpedSiameseRNN(hyper=hyper)
    elif config.architecture_type == 'SiameseCNN':
        model = cnn_models.CNNSim(hyper=hyper)
    elif config.architecture_type == 'WarpedSiameseCNN':
        model = cnn_models.CNNWarpedSim(hyper=hyper)

    # create a model (see sim_model.py) --> encoder, sim, optimizer routine
    model.create_model()
    print("Model with", model.num_model_parameters(), "parameters created", '\n')

    # create an optimizer and start the learning process
    opt = Optimizer(hyper=hyper, dataset=dataset, sim_model=model)
    opt.optimize()


if __name__ == '__main__':
    main()
