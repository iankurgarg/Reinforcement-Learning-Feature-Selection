import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pandas as pd

# This script is for training the neural network and saving the output files in the /data folder.
# Please discretize the data and then pass to the MDP_process2.py to get the ECR values.
# P.S. Could take a lot of time to train the net and so we already have the output file descritized and ready for exeution.

temp = pd.read_csv('data/MDP_Original_data2.csv')
vec = temp.columns
temp = temp[vec[6:]]
temp = preprocessing.scale(temp)
temp.to_csv('data/scaled_124_features.csv',index=False)

temp = pd.read_csv('data/scaled_124_features.csv')
data_x = np.array(temp)
print data_x.shape

x = tf.placeholder('float',[None,124])
y = tf.placeholder('float',[None,124])

n_nodes_hl1 = 11
n_nodes_hl2 = 8
n_nodes_output = 124
#tf.set_random_seed(784)
def neural_network_model(x):
	hidden_layer1 = { 'weights' : tf.Variable(tf.random_normal([124,n_nodes_hl1])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl1])) }
	hidden_layer2 = { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases' : tf.Variable(tf.random_normal([n_nodes_hl2])) }
	output_layer = { 'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_output])), 'biases' : tf.Variable(tf.random_normal([n_nodes_output])) }
	l1 = tf.add( tf.matmul(x,hidden_layer1['weights'] ) , hidden_layer1['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add( tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['biases'])
	l2_activated = tf.nn.sigmoid(l2)
	output = tf.matmul(l2_activated, output_layer['weights']) + output_layer['biases']
	return [output,l2]

prediction = neural_network_model(x)
cost = tf.reduce_sum(tf.abs(prediction[0]-data_x))
type(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

# change the number of epochs here and train on the cluster. The file will automatically save the output to a csv file.
# open the output file and the scaled_124_features file and manually compare the two.
num_epochs = input('enter number of epochs:')
err = []
cont = 'y'
while cont == 'y':
	for i in range(int(num_epochs)):
		opt,c,p = session.run([optimizer,cost,prediction], feed_dict={x:data_x})
		if i%100 == 0:
			print c
			err.append(c)
	cont = input('continue training? (y/n):')
	if cont == 'y':
		num_epochs = input('enter number of epochs:')
	else:
		print 'training completed'

p = session.run(prediction, feed_dict={x:data_x})
write_data = input('Save output files? say y or n:')
if write_data == 'y':
	l20 = p[1]
	df_l2 = pd.DataFrame(l20)
	df_l2.to_csv('data/nn_layer2_output.csv',index=False)

	flo = p[0]
	df_flo = pd.DataFrame(flo)
	df_flo.to_csv('data/nn_final_layer_output.csv',index=False)

	df_err = pd.DataFrame(np.array(err))
	df_err.to_csv('data/nn_scrap/error.csv')
	print 'files saved and job complete'
else:
	print 'file NOT saved and job complete'