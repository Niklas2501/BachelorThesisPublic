import tensorflow as tf


# an abstract similarity model
class AbstractSimModel:

    def __init__(self, hyper):
        # the configuration
        self.hyper = hyper
        # define a minus one constant, not used currently
        self.minus_one_constant = tf.constant(-1.0, dtype=tf.float32)
        # the maximum sequence length
        self.sequence_length = self.hyper['uniwarp:length']  # equals dataset.series_length

        # define a placeholder as input for all models
        # shape: #examples X length of time series X #channels
        # 2*batch_size because batch_size is the number of pairs
        # See also: https://bit.ly/2KXK7uE Tips Section:
        # The meaning of the 3 input dimensions are: samples, time steps, and features.
        self.input = tf.placeholder(shape=(2 * self.hyper['model:num_batch_pairs'], self.hyper['uniwarp:length'],
                                           self.hyper['dataset:num_channels']), dtype=tf.float32)

        # 1d array placeholder that will be filled if 0 if pair at index has the same class and 0 if not
        self.true_similarities = tf.placeholder(shape=(self.hyper['model:num_batch_pairs'],), dtype=tf.float32)

        # set by create_encoder, contains the distances of each pair
        self.distances_of_pairs = None

        # define the activations, similarity and update rule
        self.architecture = None
        self.loss, self.pred_similarities, self.update_rule = None, None, None

        # store the regularisation penalty from the config in a constant
        self.reg_penalty = tf.constant(self.hyper['uniwarp:lambda'], dtype=tf.float32)

        # name of the model is set to a default name
        self.name = 'AbstractSingleSimModel'
        self.is_training = tf.placeholder(tf.bool)
        self.additional_loss = None

    # count the number of parameters in the model, used to output
    def num_model_parameters(self):

        total_parameters = 0

        for variable in tf.trainable_variables():
            shape = variable.get_shape()

            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value

            total_parameters += variable_parameters

        return total_parameters

    # will be overridden by the child classes
    def create_encoder(self):
        print("ERROR: Encoder left undefined")
        pass

    # will be overridden by the child classes
    def create_similarity(self):
        print("ERROR: Similarity left undefined")
        pass

    # will be overridden by the child classes
    def calc_distance_of_pair(self, pair_ixd):
        print("ERROR: Distance of pairs left undefined")
        pass

    # this method defines how the model is updated at each iteration
    def create_optimization_routine(self):
        # create an update rule using the Adam optimizer for
        # maximizing similarity if similarity_sign == 1 and
        # minimizing similarity if similarity_sign == -1

        with tf.variable_scope("OptimizationRoutines"):
            # define the loss function
            self.loss = tf.losses.log_loss(self.true_similarities, self.pred_similarities)

            # add additional loss terms, e.g. regularization (not used currently)
            if self.additional_loss is not None:
                print("Adding penalty term", self.additional_loss)
                self.loss += self.reg_penalty * self.additional_loss

            # get all the trainable variables and default update operator
            trainable_vars = tf.trainable_variables()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                # execute the first step of the update: compute the gradients of the loss function
                # with respect to the trainable variables with tf.gradients
                # then clip the gradients by a maximum value
                clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars),
                                                          self.hyper['uniwarp:max_grad_norm'])

                # Create the updated rule that is run in update_model
                # Therefore create an adam optimizer with the configured initial learning rate
                # that applies the previously calculated and clipped gradients to the trainable variables
                self.update_rule = tf.train.AdamOptimizer(self.hyper['uniwarp:eta']).\
                    apply_gradients(zip(clipped_grads, trainable_vars))

    # create the model
    def create_model(self):

        # create the encoder
        self.create_encoder()

        # create the similarity
        self.create_similarity()

        # create the optimization nodes
        self.create_optimization_routine()
