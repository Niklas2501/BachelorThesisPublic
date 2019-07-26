import tensorflow as tf
from neural_network.sim_model import AbstractSimModel


class RNNAbstractSimModel(AbstractSimModel):

    def __init__(self, hyper):
        # initialise the super class
        AbstractSimModel.__init__(self, hyper)
        self.architecture = None
        self.cells = None
        self.distances_of_pairs = None
        self.name = 'RNNSingleAbstractSimModel'

    def create_encoder(self):
        with tf.variable_scope("RNNEncoder"):

            print('Creating RNN encoder')
            cells_list = []  # list that contains the forward-facing recurrent layers

            # Array length determines the number of layers, the entrys the number of cells in the layer
            for num_cells in self.hyper['uniwarp:rnn_encoder_layers']:
                # Create a new layer with num_cells units and a tanh activation function
                # Mind the naming: a LSTMCell is the layer
                # units is not the amount of neurons or cells in this layer
                # units only define the size of the cell state and output vector
                print("RNN layer with a", num_cells, "unit output vector")
                cells_list.append(tf.keras.layers.LSTMCell(units=num_cells, activation=tf.keras.activations.tanh))

            self.cells = tf.keras.layers.StackedRNNCells(cells_list)
            rnn = tf.keras.layers.RNN(self.cells, return_sequences=True)
            bi = tf.keras.layers.Bidirectional(layer=rnn)(self.input)
            batch_norm = tf.keras.layers.BatchNormalization()(bi)
            dropout = tf.keras.layers.Dropout(rate=self.hyper['uniwarp:dropout_rate'])(batch_norm)
            self.architecture = dropout


# the siamese rnn model
class SiameseRNN(RNNAbstractSimModel):

    # constructor
    def __init__(self, hyper):
        RNNAbstractSimModel.__init__(self, hyper)  # call super constructor
        self.name = 'SiameseRNN'  # rename model

    def create_encoder(self):
        RNNAbstractSimModel.create_encoder(self)  # Create a bidirectional rnn that is used as sub network

        with tf.variable_scope("SiameseRNNEncoder"):
            # distances_of_pairs = tensor containing the distances between all pairs
            # applies the distance between a pair function to each pair in a batch
            self.distances_of_pairs = tf.map_fn(lambda pair_index: self.calc_distance_of_pair(pair_index),
                                                # Distance between a single pair
                                                # a tesor with content like normal range function
                                                tf.range(self.hyper['model:num_batch_pairs'], dtype=tf.int32),
                                                back_prop=True,
                                                name='PairWiseDistMap',
                                                dtype=tf.float32)

    def calc_distance_of_pair(self, pair_index):
        # calc_distance_of_pair = distance between a single pair
        # Calculates the absolute difference between a pair of time series
        # (not the indented use of the losses.absolute_difference function --> label, prediction)
        # for a description of the self.h structure see below
        # calculates the absolute difference between the values at each time stamp and sums everything
        return tf.losses.absolute_difference(self.architecture[2 * pair_index, :, :],
                                             self.architecture[2 * pair_index + 1, :, :])

    def create_similarity(self):
        with tf.variable_scope("SiameseRNNSimilarity"):
            # transform distance into similarity
            # self.pair_dists is a tensor containing the distances between all pairs
            # exp is element wise
            self.pred_similarities = tf.exp(-self.distances_of_pairs, name='SiameseRNNSim')


# the warped siamese rnn model
class WarpedSiameseRNN(SiameseRNN):
    # constructor
    def __init__(self, hyper):
        SiameseRNN.__init__(self, hyper)  # call super constructor
        self.name = 'WarpedSiameseRNN'
        self.is_first_call = True

    # redefine the distance between a single pair,
    # distance between all and transformation into similarity stay the same
    def calc_distance_of_pair(self, pair_index):

        with tf.variable_scope("WarpedSiameseRNNDistPair") as scope:

            # unless it is the first call, then reuse the variables of the scope
            if self.is_first_call:
                self.is_first_call = False
            else:
                scope.reuse_variables()

            # self.architecture shape: length of time series x 2*batch_size x length of rnn output
            # A and B are the two examples that form the pair with  a shape of T x K each,
            # T the time indices and K the length of the context vecotr = num_units of the last LSMT-layer, see paper

            with tf.device('/device:GPU:1'):
                a = self.architecture[2 * pair_index, :, :]
                # create an array with the indices of the times series
                indices_a = tf.range(a.shape[0])

                # replicate idx_A  A.Shape[0] times,
                # [] because it must be 1d
                # result: [0,0,0,...,0,1,1,1,...,1,2,2,2,...,A.shape[0],A.shape[0],A.shape[0],...]
                indices_a = tf.tile(indices_a, [a.shape[0]])

                # gather the features for the indices
                # values of A at index i are set to all positions where i is the value in idx_A
                a_expanded = tf.gather(a, indices_a)

            with tf.device('/device:GPU:0'):
                b = self.architecture[2 * pair_index + 1, :, :]
                indices_b = tf.range(b.shape[0])
                indices_b = tf.reshape(indices_b, [-1, 1])
                indices_b = tf.tile(indices_b, [1, b.shape[0]])

                # result: [0,1,2,...,B.shape[0],0,1,2,...,B.shape[0],0,1,2,...]
                indices_b = tf.reshape(indices_b, [-1])
                b_expanded = tf.gather(b, indices_b)

            with tf.device('/device:GPU:1'):

                # original comment and variable name misleading
                # like the paper stated this calculates the absolute distance (L1 norm)
                # as first part of the overall distance
                # Keep in mind: subtracted before
                abs_distance = tf.abs(tf.subtract(a_expanded, b_expanded))

                # Minimize the squared euclidean distance of all pairs
                # Then reformat the tensor to be compatible with the output of the wrapper function
                smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1,
                                                         name='PairsDists')

            with tf.device('/device:GPU:0'):

                # concatenate the two feature tensor to serve as the input for the warping weight neural network
                # axis = 1 --> first dimension still length of T, concatinated by dimension of sim values
                # Size: time series length ^ 2 -> Contains the concatinated feature vectors of all index combinations
                ffnn = tf.concat([a_expanded, b_expanded], axis=1, name='ConcatenatedPairwiseIndices')

                # define the warping neural network with the concatenated examples as input
                for num_units in self.hyper['uniwarp:warp_nn_layers']:
                    print('Adding Warping NN layer with ', num_units, 'neurons')
                    # Mind the variable overwriting --> added after each other
                    # Hidden units use relu as activation function
                    ffnn = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(ffnn)

                # a final linear layer for the warping weights output in [0, 1]
                ffnn = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(ffnn)

            with tf.device('/device:GPU:1'):
                # The result of the pair distance is multiplied by the result of the wrapper function
                warped_dists = tf.multiply(smallest_abs_difference, ffnn, name="WarpedSiameseRNN")

            # This is then minimized again
            return tf.reduce_mean(warped_dists)
