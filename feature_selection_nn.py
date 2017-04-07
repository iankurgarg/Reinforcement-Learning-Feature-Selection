import tensorflow as tf
import pandas as pd
import numpy as np

original_data = pd.read_csv('MDP_Original_data2.csv')
vec = list(original_data.columns)
original_data = original_data[vec[6:]]
length = float(len(original_data))
x = tf.placeholder('float',[None,124])
y = tf.placeholder('float',[None,124])

n_nodes_hl1 = 30
n_nodes_hl2 = 8
n_nodes_output = 124

data_x = np.array(original_data)
data_x.shape

# def getTrainingXY(start, batch_size=100):
# 	alpha = np.array(original_data[batch_size*start:batch_size*(start+1)])
# 	print alpha.shape
# 	return alpha, alpha

def neural_network_model(data):
	hidden_layer1 = { 'weights' : tf.Variable(tf.random_normal([124,n_nodes_hl1])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1])) }
	hidden_layer2 = { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl2])) }
	output_layer = { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_output])), 'biases' : tf.Variable(tf.random_normal([n_nodes_output])) }
	l1 = tf.add( tf.matmul(data,hidden_layer1['weights'] ) , hidden_layer1['biases'])
	l1 = tf.nn.tanh(l1)
	l2 = tf.add( tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['biases'])
	l2 = tf.nn.tanh(l2)
	output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
	return [output,l2]

# def train_model(x):
# 	prediction = neural_network_model(x)
# 	cost = tf.reduce_mean(tf.abs(prediction-y))
# 	# cost = tf.reduce_sum( tf.divide( tf.pow((prediction-y), tf.constant(2.0)), tf.constant(length)) )
# 	optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
# 	saver = tf.train.Saver()
# 	batch_size = 200
# 	num_epochs = 10
# 	with tf.Session() as sess:
# 		sess.run(tf.global_variables_initializer())
# 		for epoch in range(num_epochs):
# 			epoch_loss = 0
# 			_, c = sess.run([optimizer, cost], feed_dict = {x:data_x, y: data_x})
# 			epoch_loss = c
# 			# epoch_loss = 0
# 			# for _ in range(len(original_data)/batch_size):
# 			# 	epoch_x, epoch_y = getTrainingXY(epoch, batch_size)
# 			# 	_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
# 			# 	epoch_loss += c
# 			print 'Epoch',epoch,'completed out of',num_epochs, 'loss:', epoch_loss
# 		saver.save(sess,'saved_neural_net')
# 		print sess.run(prediction,feed_dict={x:data_x[0].reshape(1,124)})


output_df = pd.DataFrame()
prediction = neural_network_model(x)
cost = tf.abs(prediction[0]-y)
# cost = tf.reduce_sum( tf.divide( tf.pow((prediction-y), tf.constant(2.0)), tf.constant(length)) )
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
saver = tf.train.Saver()
batch_size = 200
num_epochs = 100
sess =  tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(num_epochs):
	epoch_loss = 0
	_, c = sess.run([optimizer, cost], feed_dict = {x:data_x, y: data_x})
	epoch_loss = c
	print 'Epoch',epoch,'completed out of',num_epochs, 'loss:', epoch_loss.mean()
#saver.save(sess,'saved_neural_net')

var = sess.run(prediction,feed_dict={x:data_x[0].reshape(1,124)})
var10 = sess.run(prediction,feed_dict={x:data_x[10].reshape(1,124)})
var == var10

output_df = pd.DataFrame(var)
output_df.head()

train_model(x)


saver = tf.train.Saver({'l1': l1, 'l2': l2})
test_sample = data_x[0].reshape(1,124)

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, 'saved_neural_net')
	print 'net loaded'
print done
#sess.run([])
