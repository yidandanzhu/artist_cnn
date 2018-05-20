# Artists Classification

### 1. Data Preprocessing
I separate the dataset to two parts: one is for training and validating, another one is for testing. In **dataset_prep.py** I just randomly separate the whole dataset to two different directories given a test/train ratio. So this file will be ran before the model being trained.

Each image is resized to the same shape after being read into memory, so that:
* Makes the input to the network more convenient;
* Remove the effect of the image size to the model;
* Decrease the input size and guarantee the whole dataset fit into the memory.

### 2. Model Creation
I use a CNN to be the model. The architecture is:

 **images -> convolutional layer -> convolutional layer -> convolutional layer -> fully connected layer -> fully connected layer -> output**

 Firstly, I feed the images to one convolutional layer, followed by another two convolutional layers. Then I flatten the output of convolutional layers and feed the output to two fully connected layers. And finally, the output of fully connected layers will represent the probabilities for classes into which the images will be classified as.


### 3. Hyperparameter Tuning
The parameters in the model:
* **filter_size, num_filter** in the 1st convolutional layer;
* **filter_size, num_filter** in the 2nd convolutional layer;
* **filter_size, num_filter** in the 3rd convolutional layer;
* **layer_size** in the 1st fully connected layer
* **layer_size** in the 2nd fully connected layer
* **batch_size** during the training
* **learning rate** learning rate of optimizer

The procedure I followed to tune above hyperparameters is as follow:
1. Make a list of experimental candidates for each of the hyperparameters;
2. For each hyperparameter, train the model using the each parameters in the list, with other hyperparameters fixed;
3. Compare validation set accuracy for all of the parameters in the list, choose the one with the highest validation accuracy.


### 4. Parameter Optimization
I optimized parameters: number of filters for each of nerual net layer, fully connected layer size, batch size and learning rate. I also fixed the epoch to be 15.

optimized parameters: 
	num_filters_conv1 = 16, 
	num_filters_conv2 = 8, 
	num_filters_conv3 = 8, 
	fc_layer_size = 32, 
	batch_size = 32, 
	learning_rate = 1e-4


### 5. installation requirement and command
pacakge used: tensorflow, opencv, numpy
python version: python 3
0. Make sure the dataset folder is in the same directory as the following three python files
1. prepare datasets: python dataset_prep.py 
2. train model: python artist_train.py
3. test: python artist_predict.py


### 6. Accuracy
training(last batch): 0.71875
validation: 0.77987
testing: 75.8793
