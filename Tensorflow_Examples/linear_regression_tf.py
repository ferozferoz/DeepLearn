import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
housing_data = fetch_california_housing()
m,n = housing_data.data.shape

housing_data_bias = np.c_[np.ones((m,1)),housing_data.data]
X = tf.constant(housing_data_bias,dtype=tf.float32,name="X")
y = tf.constant(housing_data.target.reshape(-1,1),dtype=tf.float32,name="y")
XT = tf.transpose(X)
""" (XT * X )-1 * X * y """

theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)
with tf.Session() as sess:
    theta_val = theta.eval()
print(theta_val)