import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

#data_x = np.array([[10,20,25,1240],[10,20,99,1240],[17,50,10,60],[10,50,30,60],[19,50,70,60]])

temp = pd.read_csv('data/scaled124_generated.csv')
data_x = np.array(temp)
print data_x.shape

x = tf.placeholder('float',[None,124])
y = tf.placeholder('float',[None,124])

n_nodes_hl1 = 10
n_nodes_hl2 = 8
n_nodes_output = 124
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
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

# change the number of epochs here and train on the cluster. The file will automatically save the output to a csv file.
# open the output file and the scaled_124_features file and manually compare the two.
num_epochs = input('enter number of epochs:')
for i in range(int(num_epochs)):
	opt,c,p = session.run([optimizer,cost,prediction], feed_dict={x:data_x})
	print c

p = session.run(prediction, feed_dict={x:data_x})
l20 = p[1]
df_l2 = pd.DataFrame(l20)
df_l2.to_csv('data/nn_layer2_output.csv',index=False)

flo = p[0]
df_flo = pd.DataFrame(flo)
df_flo.to_csv('data/nn_final_layer_output.csv',index=False)
# q = np.array(p)
# q = q.reshape(7168,124)
# df = pd.DataFrame(q)
# df.to_csv('data/nn_output10000_80_relu_124.csv',index=False)
