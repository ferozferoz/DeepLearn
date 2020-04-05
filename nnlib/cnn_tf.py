import h5py
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

class CNNModelWithTensorFlow():

    def _load_dataset(self, train_path, test_path):
        train_dataset = h5py.File(train_path, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
        test_dataset = h5py.File(test_path, "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
        classes = np.array(test_dataset["list_classes"][:])  # the list of classes
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def _convert_to_one_hot(self, Y, C):
        Y = np.eye(C)[Y.reshape(-1)]
        return Y

    def _create_placeholder(self,n_H0, n_W0, n_C0, n_y):
        # H0,W0,C0 height, width, channel
        # ny number of classes

        X = tf.placeholder(tf.float32,[None,n_H0, n_W0, n_C0])
        y = tf.placeholder(tf.float32, [None,n_y])
        return X,y

    def _initialize_parameters(self):
        tf.set_random_seed(1)
        W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

        parameters = {"W1":W1,
                      "W2":W2}
        return parameters

    def _forward_propagation(self,X,parameters):
        W1 = parameters['W1']
        W2 = parameters['W2']
        #conv2D
        Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
        #relu function
        A1 = tf.nn.relu(Z1)
        #MAXPOOL
        P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')
        Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        p = tf.contrib.layers.flatten(P2)
        Z3 = tf.contrib.layers.fully_connected(p, 6, activation_fn=None)
        return Z3

    def _compute_cost(self,Z3,Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
        return cost

    def _random_mini_batches(self,X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)
        Arguments:
        X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
        mini_batch_size - size of the mini-batches, integer
        seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        m = X.shape[0]  # number of training examples
        mini_batches = []
        np.random.seed(seed)
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :, :, :]
        shuffled_Y = Y[permutation, :]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def _model(self,X_train,Y_train,X_test,Y_test,learning_rate=.009,number_of_epochs=100,mini_batch_size=64):
        seed=3
        (m,n_H0, n_W0, n_C0) = X_train.shape
        n_y = Y_train.shape[1]
        X,y = self._create_placeholder(n_H0, n_W0, n_C0, n_y)
        parameters = self._initialize_parameters()
        Z3 = self._forward_propagation(X,parameters)
        cost = self._compute_cost(Z3,y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            num_mini_batches = int(m / mini_batch_size)
            costs=[]
            for epoch in range(number_of_epochs):
                mini_batch_cost = 0
                seed = seed+1
                mini_batches = self._random_mini_batches(X_train,Y_train,mini_batch_size,seed)

                for batch in mini_batches:
                    x_mini_batch,y_mini_batch = batch
                    _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:x_mini_batch,y:y_mini_batch})
                    mini_batch_cost +=temp_cost/num_mini_batches
                    # Print the cost every epoch
                if epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, mini_batch_cost))
                if epoch % 1 == 0:
                    costs.append(mini_batch_cost)
                    # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
        return parameters








