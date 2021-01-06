import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import matplotlib.pyplot as plt
from astropy.table import Table

batch_size = 128
test_size = 256

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, p_keep_conv, p_keep_hidden):
	#Using 1 convolutional layer
	w = init_weights([3, 3, 3, 32])							# 3x3x3 conv, 32 outputs
	w_fc = init_weights([32 * 16 * 16, 625])				# FC 32 * 16 * 16 inputs, 625 outputs
	w_o = init_weights([625, 10])							# FC 625 inputs, 10 outputs (labels)
	
	l1a = tf.nn.relu(tf.nn.conv2d(X, w,	strides=[1, 1, 1, 1], padding='SAME')) # l1a shape=(?, 32, 32, 32)
	l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],			  # l1 shape=(?, 16, 16, 32)
						strides=[1, 2, 2, 1], padding='SAME')
	l1 = tf.nn.dropout(l1, p_keep_conv)


	l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])	# reshape to (?, 16x16x32)

	l4 = tf.nn.relu(tf.matmul(l3, w_fc))
	l4 = tf.nn.dropout(l4, p_keep_hidden)

	pyx = tf.matmul(l4, w_o)
	return pyx
	
def model2(X, p_keep_conv, p_keep_hidden):
	#Using 2 convolutional layers
    w = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    w_1 = init_weights([3, 3, 32, 32])  # 3x3x32 conv, 32 outputs
    w_fc = init_weights([32 * 8 * 8, 625])  # FC 32 * 8 * 8 inputs, 625 outputs
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_1,  # l2a shape=(?, 16, 16, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
	
def model3(X, p_keep_conv, p_keep_hidden):
	#Using 3 convolutional layers
    w_1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs

    w_2 = init_weights([3, 3, 32, 32])  # 3x3x32 conv, 32 outputs

    w_3 = init_weights([3, 3, 32, 32])  # 3x3x32 conv, 32 outputs

    w_fc = init_weights([32 * 8 * 8, 625])  # FC 32 * 8 * 8 inputs, 625 outputs
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w_1,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
								  
    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w_2,  #l2a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
	
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, 
                                 strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1],
                            strides=[1,2,2,1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    pyx = tf.matmul(l4, w_o)
    return pyx
	
def model4(X, p_keep_conv, p_keep_hidden):
	#Using 2 convolutional layers, increasing size of feature maps
    w = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs
    w_1 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
    w_fc = init_weights([64 * 8 * 8, 625])  # FC 32 * 8 * 8 inputs, 625 outputs
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 16, 16, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_1,  # l2a shape=(?, 16, 16, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
	
def model5(X, p_keep_conv, p_keep_hidden):
	#Using 3 convolutional layers, increasing size of feature maps
    w_1 = init_weights([3, 3, 3, 32])  # 3x3x3 conv, 32 outputs

    w_2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 32 outputs

    w_3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 32 outputs

    w_fc = init_weights([128 * 8 * 8, 625])  # FC 32 * 8 * 8 inputs, 625 outputs
    w_o = init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)

    l1a = tf.nn.relu(tf.nn.conv2d(X, w_1,  # l1a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
								  
    l2a = tf.nn.relu(tf.nn.conv2d(l1a, w_2,  #l2a shape=(?, 32, 32, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 8, 8, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
	
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, 
                                 strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1],
                            strides=[1,2,2,1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])  # reshape to (?, 14x14x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    pyx = tf.matmul(l4, w_o)
    return pyx
	

#Changed here to use CIFAR data; these images are 32*32*3
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
trX, trY, teX, teY = train_images, train_labels, test_images, test_labels
trX = trX/255
teX = teX/255

trY = utils.to_categorical(trY)
teY = utils.to_categorical(teY)

trX = trX.reshape(-1, 32, 32, 3)  # 32*32*3 input img
teX = teX.reshape(-1, 32, 32, 3)  # 32*32*3 input img

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])



p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_array = []
py_array.append(model(X, p_keep_conv, p_keep_hidden))
py_array.append(model2(X, p_keep_conv, p_keep_hidden))
py_array.append(model3(X, p_keep_conv, p_keep_hidden))
py_array.append(model4(X, p_keep_conv, p_keep_hidden))
py_array.append(model5(X, p_keep_conv, p_keep_hidden))

accs = [[],[],[],[],[]]
num_iters = 5
num_epochs = 20

for k in range(num_iters):
	for j in range(len(py_array)):
		py_x = py_array[j]

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
		train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
		predict_op = tf.argmax(py_x, 1)

		# Launch the graph in a session
		
		with tf.Session() as sess:
			
			# you need to initialize all variables
			tf.global_variables_initializer().run()
			
			for i in range(num_epochs):
				training_batch = zip(range(0, len(trX), batch_size),
									 range(batch_size, len(trX)+1, batch_size))
				for start, end in training_batch:
					sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
												  p_keep_conv: 0.8, p_keep_hidden: 0.5})

				test_indices = np.arange(len(teX)) # Get A Test Batch
				np.random.shuffle(test_indices)
				test_indices = test_indices[0:test_size]
				
				#These lines were used to print the initial accuracies prior to graphing (for questions 2/3)
				# print("Epoch",i, ":", np.mean(np.argmax(teY[test_indices], axis=1) ==
								 # sess.run(predict_op, feed_dict={X: teX[test_indices],
																 # p_keep_conv: 1.0,
																 # p_keep_hidden: 1.0})))
				if k==0:
					accs[j].append(np.mean(np.argmax(teY[test_indices], axis=1) ==
								 sess.run(predict_op, feed_dict={X: teX[test_indices],
																 p_keep_conv: 1.0,
																 p_keep_hidden: 1.0})))
				else:
					accs[j][i] += (np.mean(np.argmax(teY[test_indices], axis=1) ==
								 sess.run(predict_op, feed_dict={X: teX[test_indices],
																 p_keep_conv: 1.0,
																 p_keep_hidden: 1.0})))
		
	print("Finished run", k+1)

accs = (np.array(accs)/num_iters)

t = Table(names=('Epoch', 'M1 Accuracy', 'M2 Accuracy', 'M3 Accuracy', 'M4 Accuracy', 'M5 Accuracy'))
for val in range(num_epochs):
		t.add_row((val, accs[0][val], accs[1][val], accs[2][val], accs[3][val], accs[4][val]))
print(t)

plt.plot(range(num_epochs), accs[0], label='Model 1', marker='o')
plt.plot(range(num_epochs), accs[1], label='Model 2', marker='o')
plt.plot(range(num_epochs), accs[2], label='Model 3', marker='o')
plt.plot(range(num_epochs), accs[3], label='Model 4', marker='o')
plt.plot(range(num_epochs), accs[4], label='Model 5', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()