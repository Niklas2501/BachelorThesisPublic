from datetime import datetime

import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration


class Optimizer:

    def __init__(self, hyper, dataset, sim_model):

        self.hyper = hyper
        self.dataset = dataset
        self.num_epochs = self.hyper['optimizer:num_epochs']
        self.sim_model = sim_model
        # create a saver
        self.saver = tf.train.Saver(max_to_keep=300)

    def optimize(self):

        # save the hyper_parameters before the optimization
        # with open("./saved_models/" + self.sim_model.name + "_hyper_params.json", "w") as hyper_params_file:
        #    json.dump(self.config, hyper_params_file)

        c = tf.ConfigProto()
        c.log_device_placement = False

        with tf.Session(config=c).as_default() as sess:

            current_epoch = 0
            # config = Configuration()
            # if config.continue_training:
            #     print('Continuing training from', config.model_file)
            #     current_epoch += int(config.model_file.split('-')[-1]) * 100 + 1
            #     # restore the model with the given name (parameter when executing)
            #     self.saver.restore(sess, config.model_file)

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            loss = 0
            freq = 100

            # iterate for a number of epochs
            for epoch_index in range(current_epoch, self.num_epochs):

                batch_true_similarities = []
                batch_pairs_indices = []

                # compose batch
                # // 2 because each iteration one similar and one dissimilar pair is added
                for i in range(self.hyper['model:num_batch_pairs'] // 2):
                    pos_pair = self.dataset.draw_pair(True)
                    batch_pairs_indices.append(pos_pair[0])
                    batch_pairs_indices.append(pos_pair[1])
                    batch_true_similarities.append(1.0)

                    neg_pair = self.dataset.draw_pair(False)
                    batch_pairs_indices.append(neg_pair[0])
                    batch_pairs_indices.append(neg_pair[1])
                    batch_true_similarities.append(0.0)

                # execute the update
                pair_loss = self.update_model(sess, batch_pairs_indices, batch_true_similarities)

                # add the loss of this batch to the global loss
                loss += pair_loss

                if epoch_index % freq == 0:

                    if epoch_index > 0:
                        loss /= freq  # why? calculates the mean loss over the last 100 epochs?

                    print('Timestamp:', datetime.now().strftime('%d %H:%M:%S'), '\tEpoch:', epoch_index, '\tLoss:',
                          loss)

                    config = Configuration()
                    self.saver.save(sess, config.models_folder + self.sim_model.name + "_" + self.dataset.dataset_name
                                    + ".ckpt", global_step=epoch_index // freq)

                    loss = 0  # reset the loss for the next 100 epochs

    # update the model for the pairs of similar and dissimilar series
    def update_model(self, sess, batch_pairs_idxs, batch_true_similarities):

        # create a batch of examples with the given indices, change the list of ground truth to an array
        model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_idxs, axis=0)
        sim_batch = np.asarray(batch_true_similarities)

        # compute loss of the loss function defined in the create_optimization_routine method
        # only done to get the loss value, is calculated again when executing the update rule in the next step
        pair_loss = sess.run(self.sim_model.loss, feed_dict={
            self.sim_model.input: model_input,
            self.sim_model.true_similarities: sim_batch,
            self.sim_model.is_training: False})

        # execute the update rule defined in the create_optimization_routine method
        sess.run(self.sim_model.update_rule,
                 feed_dict={self.sim_model.input: model_input,
                            self.sim_model.true_similarities: sim_batch,
                            self.sim_model.is_training: True})

        # return the loss of this batch
        return pair_loss
