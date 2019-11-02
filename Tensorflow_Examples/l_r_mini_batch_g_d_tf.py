import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import math
learning_rate = 0.001
n_epoch = 1000
batch_size = 100
housing = fetch_california_housing()
m,n = housing.data.shape
n_batches = math.floor(m/batch_size)
housing_bias = np.c_[np.ones((m,1)),housing.data]
housing_target = housing.target.reshape(-1,1)
X = tf.placeholder(tf.float32,shape=(None,n+1),name="X")
y = tf.placeholder(tf.float32,shape=(None,1),name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X,theta,name="prediction")
y_pred = tf.sigmoid(y_pred)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
#gradient = 1/m * tf.matmul(tf.transpose(X),error)
#training_op = tf.assign(theta, theta - learning_rate * gradient )
def fetch_data(batch_index,batch_size):
    k = batch_index*batch_size
    l = (batch_index+1)* batch_size
    X_batch = housing_bias[k:l,]
    y_batch = housing_target[k:l,]
    return  X_batch,y_batch

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        for batch_index in range(n_batches):
            X_batch,y_batch = fetch_data(batch_index,batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
    best_theta = theta.eval()
    print(best_theta)