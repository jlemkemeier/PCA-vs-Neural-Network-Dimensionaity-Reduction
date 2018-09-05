
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import linalg as LA
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import soundfile as sf


#import songs, and divide by their vector norm
for x in range(1, 101):
    sig, samplerate = sf.read('/Users/jlemkemeier/Desktop/Cuts/' + str(x) +'c.wav')
    sig = np.array(sig)

    #unify all audio clips to have two values per unit of time
    try:
        if(len(sig[0]) == 2):
            sig = np.sum(sig, axis=1)/2
    except:
        print("not 2")

    #dived each value in the audio clip by the vector norm
    for z in range(len(sig)):
        if z == 0:
            newsig = np.array([sig[z]/LA.norm(sig, np.inf)])
        else:
            newsig = np.hstack((newsig, [sig[z]/LA.norm(sig, np.inf)]))

    if x == 1:
        total = np.array(newsig)
    else:
        total = np.vstack((total, newsig))
print(len(total))
X = total

# Python optimisation variables
epochs = 100000
learning_rate = tf.placeholder(tf.float32, shape=[])
batch_size = 25
x_size = 2000
y_size = 2000

# declare the training data placeholders
x = tf.placeholder(tf.float32, [None, x_size], name='x')
y = tf.placeholder(tf.float32, [None, y_size], name='y')
points = np.array([])

# set up parameters
W = []
b = []
h_size = [100, 50, 3, 50, 100]
layer = []

# now declare the weights connecting the input to the hidden layer
W.append(tf.Variable(tf.random_normal([x_size, h_size[0]], stddev=0.03), name='W1'))
b.append(tf.Variable(tf.random_normal([h_size[0]]), name='b1'))
layer.append(tf.nn.relu(tf.add(tf.matmul(x, W[0]), b[0])))

for i in range(1,len(h_size)):
    W.append(tf.Variable(tf.random_normal([h_size[i-1], h_size[i]], stddev=0.03), name='W2'))
    b.append(tf.Variable(tf.random_normal([h_size[i]]), name='b2'))
    layer.append(tf.nn.relu(tf.matmul(layer[i-1], W[i])+ b[i]))
    if(i == int(len(h_size)/2)):
        points = layer[-1]

W.append(tf.Variable(tf.random_normal([h_size[-1], y_size], stddev=0.03), name='W3'))
b.append(tf.Variable(tf.random_normal([y_size]), name='b3'))

y_ = tf.nn.softmax(tf.add(tf.matmul(layer[-1], W[-1]), b[-1]))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
mean_squared_error = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, y), axis=1))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_squared_error)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

finalpoints = np.array([[0, 0, 0]])

# start the session
with tf.Session() as sess:
   # initialise the variables
    sess.run(init_op)
    total_batch = int(len(X) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        iteration = 0
        for i in range(total_batch):
            batch_x = X[iteration:iteration+batch_size]
            batch_y = X[iteration:iteration+batch_size]
            iteration = iteration+batch_size
            _, c = sess.run([optimiser, mean_squared_error],
                         feed_dict={x: batch_x, y: batch_y, learning_rate: .00001})
            avg_cost += c / total_batch
        if (epoch%100 == 0):
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    for z in range(1,100):
       finalpoints = np.vstack((finalpoints, points.eval(feed_dict={x: [X[z]]})))
    print(sess.run(accuracy, feed_dict={x: X, y: X}))
print(finalpoints)


N = 50
x1 = finalpoints[1:50,0]
print(x1)
x2 = finalpoints[50:100,0]
y1 = finalpoints[1:50,1]
y2 = finalpoints[50:100,1]
z1 = finalpoints[1:50,2]
z2 = finalpoints[50:100,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, c='blue', alpha=.5)
ax.scatter(x2, y2, z2, c='red', alpha=.5)
plt.show()

fig.savefig("3D2norm10000e10lr.pdf", bbox_inches='tight')
