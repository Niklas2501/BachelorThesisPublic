import tensorflow as tf
from neural_network.sim_model import AbstractSimModel


class CNNAbstractSimModel(AbstractSimModel):

    def __init__(self, hyper):
        AbstractSimModel.__init__(self, hyper)
        self.last_feature_map = None
        self.name = 'CNNAbstractSimModel'
        self.architecture = None
        self.distances_of_pairs = None

    def create_encoder(self):
        with tf.name_scope("CNNEncoder"):
            # input placeholder
            feature_map = self.input

            # each is a array of the same length
            # each iteration means: create a new conv layer with num_filters many filters, that each
            # have a size of filter_size and that uses the given stride value
            for num_filters, filter_size, stride in zip(self.hyper['uniwarp:cnn_encoder_layers'],
                                                        self.hyper['uniwarp:cnn_kernel_lengths'],
                                                        self.hyper['uniwarp:cnn_strides']):
                # See above
                # Padding: "valid" means "no padding".
                feature_map = tf.keras.layers.Conv1D(filters=num_filters, padding='VALID', kernel_size=filter_size,
                                                     strides=stride)(feature_map)

                # Add a batch norm and a relu layer after each convolution layer
                # No max pooling layers are used
                feature_map = tf.keras.layers.BatchNormalization()(feature_map)
                feature_map = tf.nn.relu(feature_map)
                print('Add CNN layer', feature_map, 'with filter size ', filter_size)

            # pass the last feature map through drop out
            self.architecture = tf.keras.layers.Dropout(rate=self.hyper['uniwarp:dropout_rate'], name='CNNActivation')(
                feature_map)


class CNNSim(CNNAbstractSimModel):

    def __init__(self, hyper):
        CNNAbstractSimModel.__init__(self, hyper)
        self.name = 'CNNSim'

    def create_encoder(self):
        CNNAbstractSimModel.create_encoder(self)

        with tf.variable_scope("CNNEncoderDists"):
            self.distances_of_pairs = tf.map_fn(lambda pair_idx: self.calc_distance_of_pair(pair_idx),
                                                tf.range(self.hyper['model:num_batch_pairs'], dtype=tf.int32),
                                                back_prop=True,
                                                name='PairWiseDistMap',
                                                dtype=tf.float32)

    def calc_distance_of_pair(self, pair_index):
        return tf.losses.absolute_difference(self.architecture[2 * pair_index, :, :],
                                             self.architecture[2 * pair_index + 1, :, :])

    def create_similarity(self):
        with tf.variable_scope("CNNSimilarity"):
            # exp is element wise
            self.pred_similarities = tf.exp(-self.distances_of_pairs, name='CNNSim')


# the warped version based on CNN features
class CNNWarpedSim(CNNSim):

    def __init__(self, hyper):
        CNNSim.__init__(self, hyper)
        self.name = 'CNNWarpedSim'
        self.is_first_dist_pair_call = True

    # redefine the distance between a pair of instances
    def calc_distance_of_pair(self, pair_index):

        with tf.variable_scope("CNNWarpedSimDistPair") as scope:

            # unless it is the first call, then reuse the variables of the scope
            if self.is_first_dist_pair_call:
                self.is_first_dist_pair_call = False
            else:
                scope.reuse_variables()

                # self.architecture shape: length of time series x 2*batch_size x length of rnn output
                # A and B are the two examples that form the pair with  a shape of T x K each,
                # T = time indices and K = length of the context vector = num_units of the last LSMT-layer (see paper)
                a = self.architecture[2 * pair_index, :, :]
                b = self.architecture[2 * pair_index + 1, :, :]
                # print('Shape of example', a.get_shape().as_list())

                # create an array with the indices of the times series
                indices_a = tf.range(a.shape[0])

                # replicate idx_A  A.Shape[0] times,
                # [] because it must be 1d
                # result: [0,0,0,...,0,1,1,1,...,1,2,2,2,...,A.shape[0],A.shape[0],A.shape[0],...]
                indices_a = tf.tile(indices_a, [a.shape[0]])

                indices_b = tf.range(b.shape[0])
                indices_b = tf.reshape(indices_b, [-1, 1])
                indices_b = tf.tile(indices_b, [1, b.shape[0]])
                indices_b = tf.reshape(indices_b, [-1])
                # result: [0,1,2,...,B.shape[0],0,1,2,...,B.shape[0],0,1,2,...]

                # gather the features for the indices
                # values of A at index i are set to all positions where i is the value in idx_A
                a_expanded = tf.gather(a, indices_a)
                b_expanded = tf.gather(b, indices_b)

                # concatenate the two feature tensor to serve as the input for the warping weight neural network
                # axis = 1 --> first dimension still length of T, concatinated by dimension of sim values
                # Size: time series length ^ 2 -> Contains the concatinated feature vectors of all index combinations
                ab_concat = tf.concat([a_expanded, b_expanded], axis=1, name='ConcatenatedPairwiseIndices')
                # print('a ex', A_expanded.get_shape().as_list())
                # print('b ex', B_expanded.get_shape().as_list())
                # print('ab concat', AB_concat.get_shape().as_list())

                # original comment and variable name misleading
                # like the paper stated this calculates the absolute distance (L1 norm)
                # as first part of the overall distance
                # Keep in mind: subtracted before
                abs_distance = tf.abs(tf.subtract(a_expanded, b_expanded))

                # Minimize the squared euclidean distance of all pairs
                # Then reformat the tensor to be compatible with the output of the wrapper function
                smallest_abs_difference = tf.expand_dims(tf.reduce_mean(abs_distance, axis=1), axis=-1,
                                                         name='PairsDists')

                # define the warping neural network
                ffnn = ab_concat
                for num_units in self.hyper['uniwarp:warp_nn_layers']:
                    print('Adding Warping NN layer with ', num_units, 'neurons')
                    # Mind the variable overwriting --> added after each other
                    # Hidden units use relu as activation function
                    ffnn = tf.keras.layers.Dense(units=num_units, activation=tf.keras.activations.relu)(ffnn)

                # a final linear layer for the warping weights output in [0, 1]
                ffnn = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)(ffnn)

                # The result of the pair distance is multiplied by the result of the wrapper function
                warped_dists = tf.multiply(smallest_abs_difference, ffnn, name="WarpedSiameseRNN")

                # This is then minimized again
                return tf.reduce_mean(warped_dists)
