import tensorflow as tf
import numpy as np

learning_rate = 0.001
img_size = 201
feature = img_size*img_size
lenclass = 3

p_holder= tf.placeholder(tf.float32,[2,4])
# input place holders
X = tf.placeholder(tf.float32, [None, feature])
X_img = tf.reshape(X, [-1, img_size, img_size, 1])  # img 64x64x1 (black/white), Gray:1, RGB: 3
Y = tf.placeholder(tf.float32, [None, lenclass])

# Conv -> ReLU -> Max Pooling
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
#L1 = tf.nn.conv2d(X_img, W1, strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Conv -> ReLU -> Max Pooling
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully Connected Layer
L2_flat = tf.reshape(L2, [-1, 16 * 16 * 64]) # (img_size/2)/2 = 16
W3 = tf.get_variable("W3", shape=[16 * 16 * 64, lenclass], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([lenclass]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss &amp;amp;amp; optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

####### Training dataset #######
training_epochs = 15
batch_size = 100

train_input = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_input = np.load('test_data.npy')
test_label = np.load('test_label.npy')
save_file = './train_model.ckpt'
saver = tf.train.Saver()

# initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(train_input) / batch_size)

        for i in range(total_batch):
            start = ((i + 1) * batch_size) - batch_size
            end = ((i + 1) * batch_size)
            batch_xs = train_input[start:end]
            batch_ys = train_label[start:end]
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    saver.save(sess, save_file)

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: test_input, Y: test_label}))