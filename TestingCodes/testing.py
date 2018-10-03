import alphavantageAPI as av
from source import technical_indicators as ta 
import pandas as pd 
import tensorflow as tf 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

data = av.getCurrency("FX_DAILY", "EUR","USD")

# print (data)
# av.plotData(data)
# adding technical indicator to data
data = ta.moving_average(data, 5)
data = ta.moving_average(data, 15)
data = ta.moving_average(data, 30)
data = ta.relative_strength_index (data, 14)

# print (data.head(5))
data = data[["Close","Open", "MA_5", "MA_15","MA_30", "RSI_14"]]
data = data.fillna(0)
# print (data.head(5))
train_size = int (data.shape[0]*0.8)

# Make data a np.array
data = data.values
# Spliting data into training and testing data

train_data = data[0:train_size,:]
test_data = data[train_size:,:]

# scaler = MinMaxScaler(feature_range=(0,1))
# scaler.fit(train_data)
# train_data = scaler.transform(train_data)
# test_data = scaler.transform(test_data)
x_train = train_data[:,1:]
y_train = train_data[:,0]
x_test = test_data[:,1:]
y_test = test_data[:,0]



# Build a neural network with 2 layers
layer1_nodes = 20
layer2_nodes = 10
features = x_train.shape[1]
net = tf.InteractiveSession()
X = tf.placeholder(dtype = tf.float32, shape=[None, features])
Y = tf.placeholder(dtype= tf.float32, shape= [None])

# Initializers
sigma = 1
weight_init = tf.variance_scaling_initializer(mode="fan_avg", distribution = "uniform", scale = sigma)
bias_init = tf.zeros_initializer()

#Define hidden weights and bias
W_hidden1 = tf.Variable(weight_init([features,layer1_nodes]))
bias_hidden1 = tf.Variable(bias_init([layer1_nodes]))
W_hidden2 = tf.Variable(weight_init([layer1_nodes, layer2_nodes]))
bias_hidden2 = tf.Variable(bias_init([layer2_nodes]))

#Output Weights
W_out = tf.Variable(weight_init([layer2_nodes,1]))
bias_out = tf.Variable(bias_init([1]))

#Define hidden layers
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden1), bias_hidden1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden2), bias_hidden2))
out = tf.transpose(tf.add(tf.matmul(hidden_2, W_out), bias_out))
#Cost function
mse = tf.reduce_mean(tf.squared_difference(out,Y))
# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

#Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)

plt.show()
# Fit neural net
batch_size = 256
mse_train = []
mse_test = []
# Run
epochs = 10
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: x_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: x_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: x_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
plt.ioff()
plt.show()
print ("Complete")