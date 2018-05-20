import tensorflow as tf
import numpy as np
import os,cv2
import random

classes = ['picasso', 'vangogh']
num_classes = len(classes)
 
test='artist_dataset_test/'

image_size=128
num_channels=3
test_images = []

for img in os.listdir('artist_dataset_test/Picasso'):
	if img[0]!='.':
		n= cv2.imread('artist_dataset_test/Picasso/'+img)
		image = cv2.resize(n, (image_size, image_size), cv2.INTER_LINEAR)
		label = np.zeros(num_classes)
		label[0] =1
		test_images.append([image,label])

for img in os.listdir('artist_dataset_test/vanGogh/'):
	if img[0]!='.':
		n= cv2.imread('artist_dataset_test/vanGogh/'+img)
		image = cv2.resize(n, (image_size, image_size), cv2.INTER_LINEAR)
		label = np.zeros(num_classes)
		label[1] = 1
		test_images.append([image,label])

random.shuffle(test_images)

x_batches = [img.reshape(1, image_size,image_size,num_channels) for [img,l] in test_images]
y_batches =[l for [img,l] in test_images]


pred_result = []

#restore model
sess = tf.Session()
saver = tf.train.import_meta_graph('cnn_model/the_cnn_model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('cnn_model/'))
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")

#feed the test images
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros(num_classes).reshape(1,num_classes)

for i in range(len(x_batches)):
	x_batch = x_batches[i]
	y_batch = y_batches[i].reshape(1,num_classes)
	 
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	#result:[probabiliy_of_being_the_first_class probability_of_being_the_second_class]
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	#binarize result
	result = np.around(result)
	pred_result.append(np.array_equal(result, y_batch))

#test accuracy
test_accuracy = float(sum(pred_result))/float(len(pred_result))
print('Test Accuracy:', test_accuracy*100,'%')
#final test accuracy
