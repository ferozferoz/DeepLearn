import tensorflow as tf
from  sklearn.datasets import  fetch_california_housing
housing = fetch_california_housing()
print(housing.target.shape)
print(housing.target.reshape(-1,1).shape)

x = tf.Variable(3,name="x")
y = tf.Variable(3,name="y")
z = x*x + y + 2

"""
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(z)
print(result)
"""
init = tf.global_variables_initializer()
with tf.Session as sess:
    init.run()
    result = z.eval()
    print(result)