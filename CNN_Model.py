import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
#=================================================================
num_classes = 6 #Number of Present classes
img_size = 28
num_channels = 4
img_size_flat = img_size * img_size * num_channels
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#Hyperparameters
batch_size = 16
learning_rate = 0.005
epochs = 20

#This is a really useful way to initialize new conv layers, where it becomes almost modular with this helper function.
def create_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding="SAME")

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
    layer = tf.nn.relu(layer)

    return layer, weights

#Flattens the Conv layer, quite literally
def flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features


#Helper function so that we can define a fully connected layer, connected to flattened conv layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name="x")

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name="y_true")

y_true_cls = tf.argmax(y_true, axis=1)
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Convolutional Layer 3
filter_size3 = 5
num_filters3 = 56

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.
#Constructing the ConvNet

#Layer1
layer_conv1, weights_conv1 = \
    create_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
#Layer2
layer_conv2, weights_conv2 = \
    create_conv_layer(input=layer_conv1,
                    num_input_channels=num_filters1,
                    filter_size=filter_size2,
                    num_filters=num_filters2,
                    use_pooling=True)
#Layer3
layer_conv3, weights_conv3 = \
    create_conv_layer(input=layer_conv2,
                      num_input_channels=num_filters2,
                      filter_size=filter_size3,
                      num_filters=num_filters3,
                      use_pooling=True)
#Flatten the layer
layer_flat, num_features = flatten_layer(layer_conv3)

#Fully connected layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
                        num_inputs=num_features,
                        num_outputs=fc_size,
                        use_relu=True)
#Fully connected layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

#Final Layer output uses softmax
y_pred = tf.nn.softmax(layer_fc2) 
 #Taking the maximum argument as it is the most likely class predicted by the neural network
y_pred_cls = tf.argmax(y_pred, axis=1)
#Defining Softmax cross entropy loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
#Cost function definition for optimization problem
cost = tf.reduce_mean(cross_entropy)
#Using the Adam Optimizer for optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#Judging correct_prediction
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#finding accuracy of given metric
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
batch_size = 32 #batch size

train_path = "/home/rick/Desktop/TensorPop/X_train_sat6.csv"
train_label_path = "/home/rick/Desktop/TensorPop/y_train_sat6.csv"
test_path = "/home/rick/Desktop/TensorPop/X_test_sat6.csv"
test_label_path     = "/home/rick/Desktop/TensorPop/y_test_sat6.csv"

#Helper function for batching the csv file
def read_csv_data(path, label_path, batch_index):
    
    start_index = batch_size * batch_index
    imgbatch = []
    imglabel = []
    data = csv.reader(open(path))
    data_label = csv.reader(open(label_path))
    # Read the column names from the first line of the file
    for i in range (start_index):
        line = next(data)
        line1 = next(data_label)
    for l in range(batch_size):
        line = next(data)
        line1 = next(data_label)
        imgbatch.append(line)
        imglabel.append(line1)
    imgbatch = np.array(imgbatch, dtype=np.uint8)
    imglabel = np.array(imglabel, dtype=np.uint8)
    #print(len(imgbatch))
    return imgbatch, imglabel

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    def optimize(num_epochs):
        print("Train Train, lose your brain!")
        for i in range(0, num_epochs):
            val = []
            val1 = []
            for batch_index in range(0, math.ceil(0.7*(324000/batch_size))):
                X_batch, y_true_batch = read_csv_data(train_path,train_label_path,batch_index)
                feed_dict_train = {x : X_batch, y_true : y_true_batch}
                sess.run(optimizer, feed_dict=feed_dict_train)
                print("Batch:", batch_index, end="\r")
                val.append(sess.run(correct_prediction, feed_dict=feed_dict_train))

            for batch_index in range(0, math.ceil(0.7*(324000/batch_size))):
                X_batch, y_true_batch = read_csv_data(train_path, train_label_path, batch_index)
                feed_dict_train = {x : X_batch, y_true : y_true_batch}
                val.append(sess.run(correct_prediction, feed_dict=feed_dict_train))
            val_train = np.mean(np.array(val))
            print("Training Accuracy", val_train* 100)
            
            for batch_index in range(math.ceil(0.7*(324000/batch_size)), math.ceil(324000/batch_size)):
                X_batch, y_true_batch = read_csv_data(train_path, train_label_path, batch_index)
                feed_dict_validation = {x : X_batch, y_true : y_true_batch}
                val1.append(sess.run(correct_prediction, feed_dict=feed_dict_validation))
            val_acc = np.mean(np.array(val1))
            print("Validation Accuracy", val_acc * 100)
    #Function for evaluation on Test set
    def testonimg():
        test = []
        for batch_index in range(0, math.ceil(81000/batch_size)):
            X_batch, y_true_batch = read_csv_data(test_path,test_label_path, batch_index)        
            feed_dict_test = {x : X_batch, y_true : y_true_batch}   
            #sess = tf.Session()
            test.append(sess.run(correct_prediction, feed_dict=feed_dict_test))
        test_accuracy = np.mean(np.array(test))
        print("Test Accuracy:", test_accuracy * 100)

    optimize(1000)
    testonimg()
        #sess.close()
    saver.save(sess, '/home/rick/Desktop/TensorPop/model/deepsat-model')
    print("Model Saved!")
