import numpy as np
import os
from sklearn import preprocessing


class Dataset:

    def __init__(self):
        self.y_train = None
        self.x_train = None
        self.Y_train_raw = None
        self.Y_test_raw = None
        self.y_test = None
        self.x_test = None
        self.dataset_name = None
        self.num_classes = None
        self.series_length = None
        self.num_channels = None
        self.num_train_instances = None
        self.dataset_name = None
        self.num_test_instances = None
        self.num_instances = None
        self.onehot_encoder = None

    def load_multivariate(self, dataset_prefix):

        x_train = np.load(dataset_prefix + "train_features.npy")  # data for training
        y_train = np.load(dataset_prefix + "train_labels.npy")  # labels of the training data

        x_test = np.load(dataset_prefix + "test_features.npy")  # data for testing
        y_test = np.load(dataset_prefix + "test_labels.npy")  # labels of the testing data

        # transpose the arrays with the labels to a vertical vector
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)

        # length of the first array dimension is the number of examples
        self.num_train_instances = x_train.shape[0]
        self.num_test_instances = x_test.shape[0]

        # the total sum of examples
        self.num_instances = self.num_train_instances + self.num_test_instances

        # length of the second array dimension is the length of the time series
        self.series_length = x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.num_channels = x_train.shape[2]

        # get the number of classes by unique labels
        sorted_label_values = np.unique(y_train)
        self.num_classes = sorted_label_values.size

        # Use the OneHotEncoder of sklearn
        # Create a encoder, sparse output must be disabled to get the intended output format
        # Added categories='auto' to use future behavior
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # Prepare the encoder with training and test labels to ensure all are present
        # The fit-function 'learns' the encoding but does not jet transform the data
        # The axis argument specifies on which the two arrays are joined
        onehot_encoder = onehot_encoder.fit(np.concatenate((y_train, y_test), axis=0))

        # This transforms the vector of labels into a onehot matrix
        y_train_onehot = onehot_encoder.transform(y_train)
        y_test_onehot = onehot_encoder.transform(y_test)

        # Safe for inverse transformation
        self.onehot_encoder = onehot_encoder

        self.x_train, self.y_train = x_train, y_train_onehot
        self.x_test, self.y_test = x_test, y_test_onehot

        # Generate data set name, not clear what other operations are used for
        path = os.path.normpath(dataset_prefix)
        ds_path, fold = os.path.split(path)
        root, ds_name = os.path.split(ds_path)
        self.dataset_name = ds_name + "_" + fold

        # Data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: Array of all channels
        print('Shape of training set:', self.x_train.shape, 'Shape of test set:',
              self.x_test.shape, )

        # print(self.dataset_name, 'num_train_instances', self.num_train_instances, 'series_length', self.series_length,
        #      'num_channels', self.num_channels, 'num_classes', self.num_classes)

        # Labels
        # 1. dimension: example
        # 2. dimension: label
        # print('Test labels', self.y_train.shape)

    # not used currently
    def load_ucr_univariate_data(self, dataset_folder=None):

        # read the dataset name as the folder name
        self.dataset_name = os.path.basename(os.path.normpath(dataset_folder))

        # load the train and test data from files
        file_prefix = os.path.join(dataset_folder, self.dataset_name)
        train_data = np.loadtxt(file_prefix + "_TRAIN", delimiter=",")  # Other file format
        test_data = np.loadtxt(file_prefix + "_TEST", delimiter=",")

        # set train data
        self.y_train = train_data[:, 0]  # labels are the first column
        self.x_train = train_data[:, 1:]  # rest is data
        self.num_train_instances = self.x_train.shape[0]

        # get the series length
        self.series_length = self.x_train.shape[1]

        # set the test data
        self.y_test = test_data[:, 0]
        self.x_test = test_data[:, 1:]
        self.num_test_instances = self.x_test.shape[0]

        self.Y_train_raw = train_data[:, 0]
        self.Y_test_raw = test_data[:, 0]

        # the num of instances
        self.num_instances = self.num_train_instances + self.num_test_instances

        # get the label values in a sorted way
        sorted_label_values = np.unique(self.y_train)
        self.num_classes = sorted_label_values.size  # number of unique classes

        # print('Series length', self.series_length, ', Num classes', self.num_classes)

        # encode labels to a range between [0, num_classes)
        label_encoder = preprocessing.LabelEncoder()
        label_encoder = label_encoder.fit(self.y_train)
        Y_train_encoded = label_encoder.transform(self.y_train)
        Y_test_encoded = label_encoder.transform(self.y_test)

        # convert the encoded labels to a 2D array of shape (num_instances, 1)
        Y_train_encoded = Y_train_encoded.reshape(len(Y_train_encoded), 1)
        Y_test_encoded = Y_test_encoded.reshape(len(Y_test_encoded), 1)

        # one-hot encode the labels
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder = onehot_encoder.fit(Y_train_encoded)
        self.y_train = onehot_encoder.transform(Y_train_encoded)
        self.y_test = onehot_encoder.transform(Y_test_encoded)

        # normalize the time series
        X_train_norm = preprocessing.normalize(self.x_train, axis=1)
        X_test_norm = preprocessing.normalize(self.x_test, axis=1)

        self.x_train = np.expand_dims(X_train_norm, axis=-1)
        self.x_test = np.expand_dims(X_test_norm, axis=-1)

        self.num_channels = 1

        print('Train shape', self.x_train.shape)
        print('Test shape', self.x_test.shape)

    # not used currently
    # draw a random set of instances from the training set
    def draw_batch(self, batch_size):
        # draw an array of random numbers from 0 to num rows in X_train
        random_row_indices = np.random.randint(0, self.num_train_instances, size=batch_size)
        X_batch = self.x_train[random_row_indices]
        Y_batch = self.y_train[random_row_indices]
        # return in form of array
        return X_batch, Y_batch

    # not used currently
    # draw a random set of instances from the training set
    def draw_similar_batch(self, batch_size):
        # draw a random class and then draw randomly
        return None

    # not used currently
    def draw_dissimilar_batch(self, batch_size):
        # draw an array of random numbers from 0 to num rows in X_train
        random_row_indices = np.random.randint(0, self.num_train_instances, size=batch_size)
        X_batch = self.x_train[random_row_indices]
        Y_batch = self.y_train[random_row_indices]
        # slice the batch from the training set acc. to. the drawn row indices
        return X_batch, Y_batch

    # draw a random set of instances from the TRAINING SET
    def draw_pair(self, is_positive):

        # draw as long as is_positive criterion is not satisfied
        while True:

            # draw two random examples index
            first_idx = np.random.randint(0, self.num_train_instances, size=1)[0]
            second_idx = np.random.randint(0, self.num_train_instances, size=1)[0]

            # return the two indexes if they match the is_positive criterion
            if is_positive:
                if np.array_equal(self.y_train[first_idx], self.y_train[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(self.y_train[first_idx], self.y_train[second_idx]):
                    return first_idx, second_idx

    # draw a random set of instances from the TEST SET
    # see above
    def draw_test_pair(self, is_positive):

        while True:

            first_idx = np.random.randint(0, self.num_test_instances, size=1)[0]
            second_idx = np.random.randint(0, self.num_test_instances, size=1)[0]

            if is_positive:
                if np.array_equal(self.y_test[first_idx], self.y_test[second_idx]):
                    return first_idx, second_idx
            else:
                if not np.array_equal(self.y_test[first_idx], self.y_test[second_idx]):
                    return first_idx, second_idx

    # not used currently
    # retrieve the series of the pair index
    def retrieve_series_content(self, pair_index, max_length):

        # create a pair tensor filled with zeros up to max length
        X = np.zeros(shape=(2, max_length, 1))

        # set the series content from the current dataset
        series_length = self.series_length
        X[0][:series_length] = np.expand_dims(self.x_train[pair_index[0]], axis=-1)
        X[1][:series_length] = np.expand_dims(self.x_train[pair_index[1]], axis=-1)

        return X
