import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
learning_rate = 0.001
n_epoch = 2000
housing = fetch_california_housing()
m,n = housing.data.shape
housing_bias = np.c_[np.ones((m,1)),housing.data]
housing_target = housing.target.reshape(-1,1)
X = tf.constant(housing_bias,dtype= tf.float32,name="X")
y = tf.constant(housing_target,dtype= tf.float32,name="y")
theta = tf.Variable(np.random.rand(n+1,1), dtype= tf.float32, name="theta")
y_pred = tf.matmul(X, theta, name="prediction")
y_pred = tf.sigmoid(y_pred)
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
#gradient = 1/m * tf.matmul(tf.transpose(X),error)
#training_op = tf.assign(theta, theta - learning_rate * gradient )

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        if epoch%100 == 0:
            print("Mean square error after "+str(epoch)+" epochs "+ str(mse.eval()))
        sess.run(training_op)
        theta_val = theta.eval()
print(theta_val)