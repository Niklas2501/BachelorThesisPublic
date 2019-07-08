import os
import time

import numpy as np
import sklearn.metrics
import tensorflow as tf

from configuration.Configuration import Configuration
from neural_network import rnn_models, cnn_models
from neural_network.dataset import Dataset
from neural_network.hyper_parameters import HyperParams


class Inference:

    def __init__(self, model_type, model_file, dataset_path):

        self.model_type = model_type
        self.model_file = model_file
        self.dataset_path = dataset_path

        # load the configuration of the hyperparameters
        hp = HyperParams()
        self.hyper = hp.get_uniwarp_config()

        # load the dataset
        self.ds = Dataset()
        self.ds.load_multivariate(dataset_path)

        # set the length and number of channels
        self.hyper['uniwarp:length'] = self.ds.series_length
        self.hyper['dataset:num_channels'] = self.ds.num_channels

        # create the model
        self.model = None

        if model_type == 'SiameseRNN':
            self.model = rnn_models.SiameseRNN(hyper=self.hyper)
        elif model_type == 'WarpedSiameseRNN':
            self.model = rnn_models.WarpedSiameseRNN(hyper=self.hyper)
        elif model_type == 'CNNSim':
            self.model = cnn_models.CNNSim(hyper=self.hyper)
        elif model_type == 'CNNWarpedSim':
            self.model = cnn_models.CNNWarpedSim(hyper=self.hyper)
        else:
            print("Test - No model of type", model_type)

        # create a model of the selected type
        self.model.create_model()

        # a tensorflow saver to be used for loading models
        self.saver = tf.train.Saver()

        # create a batch tensor as input that will replace the placeholder
        self.input = np.zeros((2 * self.hyper['model:num_batch_pairs'], self.hyper['uniwarp:length'],
                               self.hyper['dataset:num_channels']))

        self.true_sim_batch = np.zeros((self.hyper['model:num_batch_pairs'],))

        print('The model has', self.model.num_model_parameters(), 'parameters')

    def infer_single_example(self, example: np.ndarray, sess):

        sim_all_examples = np.zeros(self.ds.num_train_instances)

        # inference is split into batch size big parts
        for index in range(0, self.ds.num_train_instances, self.hyper['model:num_batch_pairs']):

            # fix the starting index, if the batch exceeds the number of train instances
            start_index = index
            if index + self.hyper['model:num_batch_pairs'] >= self.ds.num_train_instances:
                start_index = self.ds.num_train_instances - self.hyper['model:num_batch_pairs']

            # create a batch of pair between the test series and the batch train series
            for i in range(self.hyper['model:num_batch_pairs']):
                self.input[2 * i] = example
                # print(start_index + i)
                self.input[2 * i + 1] = self.ds.x_train[start_index + i]

            # measure the similarity between the example and the training batch
            # returns the result of the tensor self.model.pred_similarities
            # --> the pairwise similarities of the input
            sim = sess.run(self.model.pred_similarities,
                           feed_dict={self.model.input: self.input})

            # collect similarities of all badges, can't be collected in simple list because
            # some sims are calculated multiple times because only full batches can be processed
            end_of_batch = start_index + self.hyper['model:num_batch_pairs']
            sim_all_examples[start_index:end_of_batch] = sim

        # return the result of the knn classifier using the calculated similarities
        return sim_all_examples

    # infer the target of the test instances of a dataset
    # starting from the {start_pct} percentage of the instances for {chunk_pct} many instances
    # e.g. start_pct=0.1, chunk_pct=0.2 means classifying the segment between [10%, 30%)
    def infer_dataset(self, start_pct, chunk_pct):

        # get the range indices
        start_range = int(start_pct * self.ds.num_test_instances)
        stop_range = int((start_pct + chunk_pct) * self.ds.num_test_instances)

        if stop_range > self.ds.num_test_instances:
            stop_range = self.ds.num_test_instances

        with tf.Session() as sess:

            # restore the model with the given name (parameter when executing)
            self.saver.restore(sess, self.model_file)

            correct, num_infers = 0, 0

            start_time = time.clock()

            # infer all examples in the given range
            for idx_test in range(start_range, stop_range):

                max_similarity = 0
                max_similarity_idx = 0

                # inference is split into batch size big parts
                for idx in range(0, self.ds.num_train_instances, self.hyper['model:num_batch_pairs']):

                    # fix the starting index, if the batch exceeds the number of train instances
                    start_idx = idx
                    if idx + self.hyper['model:num_batch_pairs'] >= self.ds.num_train_instances:
                        start_idx = self.ds.num_train_instances - self.hyper['model:num_batch_pairs']

                    # create a batch of pair between the test series and the batch train series
                    for i in range(self.hyper['model:num_batch_pairs']):
                        self.input[2 * i] = self.ds.x_test[idx_test]
                        self.input[2 * i + 1] = self.ds.x_train[start_idx + i]

                    # measure the similarity between the test series and the training batch series
                    # returns the result of the given tensor self.model.pred_similarities
                    # --> the pairwise similarities of the input X_batch
                    sim = sess.run(self.model.pred_similarities,
                                   feed_dict={self.model.input: self.input,
                                              self.model.is_training: False})

                    # check similarities of all pairs and record the index of the closest training series
                    for i in range(self.hyper['model:num_batch_pairs']):
                        if sim[i] >= max_similarity:
                            max_similarity = sim[i]
                            max_similarity_idx = start_idx + i

                # check if correctly classified
                if np.array_equal(self.ds.y_test[idx_test], self.ds.y_train[max_similarity_idx]):
                    correct += 1
                num_infers += 1

                real = self.ds.onehot_encoder.inverse_transform([self.ds.y_test[idx_test]])[0][0]
                max_sim_class = self.ds.onehot_encoder.inverse_transform([self.ds.y_train[max_similarity_idx]])[0][0]

                # print results for this batch
                # print('Example:', idx_test, '/', stop_range, 'Current accuarcy:', correct / num_infers)
                print('Example:', idx_test+1, '/', stop_range)
                print('\tClassified as:', max_sim_class, 'Correct class:', real, 'Similarity:', max_similarity)
                print('\tCorrectly classified examples:', correct)
                print('\tCurrent accuracy:', correct / num_infers)
                print('')

            elapsed_time = time.clock() - start_time

            # print complete results
            print('----------------------------------------------------')
            print('Final Result:')
            print('----------------------------------------------------')
            print('Examples classified:', num_infers)
            print('Correctly classified:', correct)
            print('Classification accuracy:', correct / num_infers)
            print('Elapsed time:', elapsed_time)
            print('----------------------------------------------------')

    # (the pairwise similarities of the test series)
    # Calculates and saves the distances of all possible pairs of the n first examples in the test set
    def test_pairwise_similarities(self, n, folder_path):

        num_test_series = n
        dists = np.zeros((num_test_series, num_test_series))

        with tf.Session() as sess:

            self.saver.restore(sess, self.model_file)

            # all pairs of the first num_test_series
            pairs_list = []
            for i in np.arange(0, num_test_series, 1):
                for j in np.arange(0, num_test_series, 1):
                    pairs_list.append((i, j))

            num_pairs = len(pairs_list)
            batch_start_pair_idx = 0

            print('Num pairs:', len(pairs_list))

            # compute pair similarities in batches
            while batch_start_pair_idx < num_pairs:

                # create a batch of pair between the test series and the batch train series
                for i in range(self.hyper['model:num_batch_pairs']):

                    # the index of the pair
                    j = batch_start_pair_idx + i
                    if j >= num_pairs:
                        j = num_pairs - 1

                    self.input[2 * i] = self.ds.x_test[pairs_list[j][0]]
                    self.input[2 * i + 1] = self.ds.x_test[pairs_list[j][1]]

                # print('batch starting at', batch_start_pair_idx)

                # measure the similarity between the test series and the training batch series
                sim = sess.run(self.model.pred_similarities,
                               feed_dict={self.model.input: self.input,
                                          self.model.is_training: False})
                # set the distances
                for i in range(self.hyper['model:num_batch_pairs']):
                    # the index of the pair
                    j = batch_start_pair_idx + i
                    if j >= num_pairs:
                        j = num_pairs - 1
                    # set the distance
                    dists[pairs_list[j][0]][pairs_list[j][1]] = 1.0 - sim[i]

                # the batch pair index increases
                batch_start_pair_idx += self.hyper['model:num_batch_pairs']

            # the distances
            print(dists.shape)

            np.save(os.path.join(folder_path, self.model.name + '_' + self.ds.dataset_name + "_dists.npy"), dists)
            np.save(os.path.join(folder_path, self.model.name + '_' + self.ds.dataset_name + "_labels.npy"),
                    self.ds.y_test[:num_test_series])

    # the pairwise similarities of the test series of num_test_batches many batches
    def pairwise_test_accuracy(self, num_test_batches):

        test_acc = 0

        with tf.Session() as sess:

            self.saver.restore(sess, self.model_file)

            for i in range(num_test_batches):

                # draw the random test batch
                batch_pairs_idxs = []
                batch_true_similarities = []

                # // because each iteration a similar and a dissimilar pair is drawn
                for j in range(self.hyper['model:num_batch_pairs'] // 2):
                    pos_idxs = self.ds.draw_test_pair(True)
                    batch_pairs_idxs.append(pos_idxs[0])
                    batch_pairs_idxs.append(pos_idxs[1])
                    batch_true_similarities.append(1.0)

                    neg_idxs = self.ds.draw_test_pair(False)
                    batch_pairs_idxs.append(neg_idxs[0])
                    batch_pairs_idxs.append(neg_idxs[1])
                    batch_true_similarities.append(0.0)

                # (the numpy tensors of the series and ground truth similarities)
                # Transform indices to examples
                X_batch = np.take(a=self.ds.x_test, indices=batch_pairs_idxs, axis=0)
                sim_batch = np.asarray(batch_true_similarities)

                # measure the batch loss of the model
                pred_similarities = sess.run(self.model.pred_similarities, feed_dict={
                    self.model.input: X_batch,
                    self.model.true_similarities: sim_batch,
                    self.model.is_training: False})

                # if sim>0.5 than it counts as the same class otherwise as different
                pred_label = np.where(pred_similarities >= 0.5, 1, 0)

                # compare the predicted with the groud right ones from sim_batch
                test_acc += sklearn.metrics.accuracy_score(sim_batch, pred_label)

                # print progress
                print(i, test_acc / (i + 1))

        # print test batches
        print(test_acc / num_test_batches)

    # the pairwise similarities of the test series
    def transductive_test_loss(self, num_test_batches):

        test_loss = 0

        with tf.Session() as sess:

            self.saver.restore(sess, self.model_file)

            # num_test_batch is a parameter given when executing
            for i in range(num_test_batches):

                # draw the random test batch
                batch_pairs_idxs = []
                batch_true_similarities = []

                # see above
                for j in range(self.hyper['model:num_batch_pairs'] // 2):
                    pos_idxs = self.ds.draw_test_pair(True)
                    batch_pairs_idxs.append(pos_idxs[0])
                    batch_pairs_idxs.append(pos_idxs[1])
                    batch_true_similarities.append(1.0)

                    neg_idxs = self.ds.draw_test_pair(False)
                    batch_pairs_idxs.append(neg_idxs[0])
                    batch_pairs_idxs.append(neg_idxs[1])
                    batch_true_similarities.append(0.0)

                # the numpy tensors of the series and ground truth similarities
                X_batch = np.take(a=self.ds.x_test, indices=batch_pairs_idxs, axis=0)
                sim_batch = np.asarray(batch_true_similarities)

                # measure the batch loss of the model
                batch_loss = sess.run(self.model.loss, feed_dict={
                    self.model.input: X_batch,
                    self.model.true_similarities: sim_batch,
                    self.model.is_training: False})

                test_loss += batch_loss

                # print progress
                print('Batch', i, ':', test_loss / (i + 1))

        # print test batches
        print('\nFinal Result:')
        print(test_loss / num_test_batches)


def main():
    # suppress debugging messages of tensorflow
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()
    model_type = config.architecture_type
    model_file = config.model_file
    dataset_path = config.preprocessed_folder
    choice = config.inference_type

    # todo remove
    # import testing.only_failure
    # dataset_path = testing.only_failure.temp_dir

    if choice == 'test':
        start_pct = 0.0
        chunk_pct = 1.0
    elif choice == 'pairwise':
        num_test_batches = 1  # Change when needed

    # prepare inference with given parameters
    ie = Inference(model_type=model_type,
                   model_file=model_file,
                   dataset_path=dataset_path)

    print('\nEnsure right model file is used:')
    print(config.model_file, '\n')

    # call the corresponding inference method
    if choice == 'test':
        ie.infer_dataset(start_pct, chunk_pct)
    elif choice == 'pairwise':
        ie.pairwise_test_accuracy(num_test_batches)


# the script for evaluating the trained neural network
if __name__ == '__main__':
    main()
