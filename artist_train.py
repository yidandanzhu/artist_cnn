import tensorflow as tf
import cv2
import os
import random
import numpy as np

def define_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def define_biases(size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

def cnn_layer(input,num_input_channels, conv_filter_size,num_filters):
        #define weight and bias
        weights = define_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = define_biases(num_filters)
        # create cnn layer
        layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
        layer += biases
        # max-pooling
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
        # fed the output of max-pooling to activation function(Relu)
        layer = tf.nn.relu(layer)
        return layer
 
def flattening_layer(layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])
        return layer


def fully_connected_layer(input, num_inputs,num_outputs,use_relu=True):
        
        #Let's define trainable weights and biases.
        weights = define_weights(shape=[num_inputs, num_outputs])
        biases = define_biases(num_outputs)
 
        layer = tf.matmul(input, weights) + biases
        if use_relu:
                layer = tf.nn.relu(layer)
 
        return layer


def learning_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,accuracy):
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0}, Training Accuracy(last batch): {1}, Validation Accuracy: {2}, Validation Loss: {3}"
        print(msg.format(epoch, acc, val_acc, val_loss))
        return msg.format(epoch, acc, val_acc, val_loss)

def get_batch(lst, start, batch_size):
    end = start+batch_size
    if end < len(lst):
        return lst[start:end]
    else:
        return lst[start:] + lst[:end%len(lst)]

def train(max_num_epoch,validation_size,optimizer,cost,accuracy,saver, batch_size):

        tstart = 0
        validation_size = int(validation_size * len(trainining_set))
        print('validation:',validation_size)

        data_validation = trainining_set[:validation_size]
        data_training =trainining_set[validation_size:]

        iteration_number = 0
        epoch = 0

        while epoch < max_num_epoch:
            
            x_batch = [img for img,l in get_batch(data_training, tstart, batch_size)]
            y_true_batch = [l for img,l in get_batch(data_training, tstart, batch_size)]
            x_valid_batch =[img for img,l in data_validation]
            y_valid_batch = [l for img,l in data_validation]
            tstart = (tstart+batch_size) % len(data_training)

            feed_dict_tr = {x: x_batch,y_true: y_true_batch}
            feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_tr)
          
            if iteration_number % int(len(data_training)/batch_size) == 0:     
                val_loss = session.run(cost, feed_dict=feed_dict_val)
                epoch +=1
                msg = learning_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,accuracy=accuracy)
                saver.save(session, 'cnn_model/the_cnn_model.ckpt')
                
            iteration_number+=1
        return msg



def cnn_model(max_num_epoch=15, filter_size_conv1 = 3, num_filters_conv1 = 16, filter_size_conv2 = 3, num_filters_conv2 = 8, filter_size_conv3 = 3, num_filters_conv3 = 8, fc_layer_size = 32, batch_size = 32, learning_rate = 1e-4):
    #neural net structure
    layer_conv1 = cnn_layer(input=x,num_input_channels=num_channels,conv_filter_size=filter_size_conv1,num_filters=num_filters_conv1)
    layer_conv2 = cnn_layer(input=layer_conv1,num_input_channels=num_filters_conv1,conv_filter_size=filter_size_conv2,num_filters=num_filters_conv2)
    layer_conv3= cnn_layer(input=layer_conv2,num_input_channels=num_filters_conv2,conv_filter_size=filter_size_conv3,num_filters=num_filters_conv3)                  
    layer_flat = flattening_layer(layer_conv3)
    layer_fc1 = fully_connected_layer(input=layer_flat,num_inputs=layer_flat.get_shape()[1:4].num_elements(),num_outputs=fc_layer_size,use_relu=True)
    layer_fc2 = fully_connected_layer(input=layer_fc1,num_inputs=fc_layer_size,num_outputs=num_classes,use_relu=False)

    #prediction
    y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
    y_pred_cls = tf.argmax(y_pred, axis=1)
    session.run(tf.global_variables_initializer())
    #calculate cost
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    #optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer()) 

    #save the model
    saver = tf.train.Saver()
    msg = train(max_num_epoch=max_num_epoch,validation_size=0.2,optimizer = optimizer,cost = cost, accuracy = accuracy, saver = saver, batch_size = batch_size)

    return msg


classes = ['picasso', 'vangogh']
num_classes = len(classes)
 
train_path='artist_dataset/'


num_channels = 3
image_size =128
images_Picasso=[]
images_vanGogh=[]

#load training,validation datasets, resize image
for img in os.listdir('artist_dataset_train/Picasso'):
    if img[0]!='.':
        n= cv2.imread('artist_dataset_train/Picasso/'+img)
        image = cv2.resize(n, (image_size, image_size), cv2.INTER_LINEAR)
        label = np.zeros(num_classes)
        label[0] =1
        images_Picasso.append([image,label])


for img in os.listdir('artist_dataset_train/vanGogh/'):
    if img[0]!='.':
        n= cv2.imread('artist_dataset_train/vanGogh/'+img)
        image = cv2.resize(n, (image_size, image_size), cv2.INTER_LINEAR)
        label = np.zeros(num_classes)
        label[1] = 1
        images_vanGogh.append([image,label])


#shuffle training,validation dataset
trainining_set = images_vanGogh+images_Picasso
random.shuffle(trainining_set)


session = tf.Session()
#placeholder for input images
x = tf.placeholder(tf.float32, shape=[None, image_size,image_size,num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


##optimize parameters

# with open('optimization_batch.txt','w') as fl:
#     for batch_size in [8,16,32,64,128]:
#         print('batch_size:',batch_size)
#         msg =cnn_model(batch_size = batch_size)
#         fl.write('batch_size:'+str(batch_size)+','+msg+'\n')

# with open('optimization_num_filters_conv1.txt','w') as fl:
#     for num_filters_conv1 in [8,16,32,64,128]:
#         print('num_filters_conv1:',num_filters_conv1)
#         msg =model_optimization(num_filters_conv1 = num_filters_conv1)
#         fl.write('num_filters_conv1:'+str(num_filters_conv1)+','+msg+'\n')

# with open('optimization_num_filters_conv2.txt','w') as fl:
#     for num_filters_conv2 in [8,16,32,64,128]:
#         print('num_filters_conv2:',num_filters_conv2)
#         msg =model_optimization(num_filters_conv2 = num_filters_conv2)
#         fl.write('num_filters_conv2:'+str(num_filters_conv2)+','+msg+'\n')


# with open('optimization_num_filters_conv3.txt','w') as fl:
#     for num_filters_conv3 in [8,16,32,64,128]:
#         print('num_filters_conv3:',num_filters_conv3)
#         msg =model_optimization(num_filters_conv3 = num_filters_conv3)
#         fl.write('num_filters_conv3:'+str(num_filters_conv3)+','+msg+'\n')


# with open('optimization_fc_layer_size.txt','w') as fl:
#     for fc_layer_size in [8,16,32,64,128]:
#         print('fc_layer_size:',fc_layer_size)
#         msg =cnn_model(fc_layer_size = fc_layer_size)
#         fl.write('fc_layer_size:'+str(fc_layer_size)+','+msg+'\n')

# with open('optimize_learning_rate.txt','w') as fl:
#     for learning_rate in [1e-4,1e-3,1e-2]:
#         print('learning_rate:',learning_rate)
#         msg =cnn_model(learning_rate = learning_rate)
#         fl.write('learning_rate:'+str(learning_rate)+','+msg+'\n')

#train model with optimized parameters, save model
cnn_model(max_num_epoch=15)

