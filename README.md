# Traffic-Sign-Interpreter

## Traffic Signal Classifier
**Final Report**

*29-06-2020*

##### **Stamatics, IIT Kanpur**


### Index


- Overview
- Sequential Model Creation
- Setting up the GUI
- Team



### Overview

Traffic Signs are an important part of the road infrastructure that provide the driver with critical information.  Hence, traffic sign recognition systems are used in autonomous vehicles etc to enable the vehicle to recognise the traffic signs put on the road. The deep learning approach uses neural networks to complete the task of traffic sign recognition.

In this project, we learnt the mathematics behind the various algorithms in Computer Vision. Finally, we implemented a Convolutional Neural Network and trained it using the readily available GTSRB dataset and evaluated it’s accuracy on the test data. Hence, our model can recognise a traffic signal and classify into different classes.


### Sequential Model Creation 

#### 1. DATASET

The CNN Model implemented by us has been trained on the GTSRB (German traffic sign detection benchmark) Dataset for traffic sign recognition. It consists of 43 different traffic sign classes and about 50,000 images. Firstly, we reshaped the images to 30 X 30 pixels size. 
Then using the inbuilt function we split the dataset into training and testing such that 20% will be used as test dataset and the rest for training the model. Following this we developed our sequential model. dataset taken from kaggle's GTSRB challenge - https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 


#### 2. CNN

Our network consists of two sets of fully connected layers and a softmax classifier. From there we begin to implement our two sets of( CONV => RELU => CONV => RELU)*2=> POOL layers.
The first set of layers uses a 5x5 kernel and Relu as the activation function to learn larger features. It will help to distinguish between different traffic sign shapes and color blobs on the traffic signs themselves. Max Pooling is done after each layer to reduce  the spatial dimensions.
Dropout is applied after each layer as a form of regularisation which aims to prevent overfitting. The result is often a more generalisable model.
Finally we add the softmax classifier in the output layer which assigns a probability to each class and hence proves to be useful in a multi-class problem such as ours.

#### 3. TRAINING AND PLOTTING 

After creating the sequential model, we initialized the optimiser and compiled the model. Following this, we plotted both loss and accuracy for the Train and Validation Data.
We obtained an accuracy of 95.19% on the Train data and an accuracy of 99.09% on the Validation data.

#### 4. MODULES USED

Basic libraries like NumPy and Matplotlib.pyplot were used. Alongside some important libraries specific to our project that have been used are:


- *Tensorflow* : Tensorflow can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. We have used it to create a sequential model.

- *Sklearn.model_selection* : The train_test_split( ) function has been used from this library to split the dataset into subsets.

- *Keras* : This library provides an interface for artificial neural networks. It acts as an interface for the Tensorflow library. We have used this library to create CNN layers.

- *Tensorflow.keras.models*  : We have imported Sequential from this library to group the stack of layers into a model.

### Setting up the GUI

tkinter is python's de-facto standard GUI package. link to tkinter documentation- https://docs.python.org/3/library/tkinter.html  . 
We had built a graphical user interface for our traffic signs classifier with Tkinter. At first we loaded the trained model ‘my_model.h5’ using Keras. (H5 is a file format to store structured data, it's not a model by itself. Keras saves models in this format as it can easily store the weights and model configuration in a single file.) And then we build the GUI for uploading the image and a button is used to classify which calls the classify() function. The classify() function is converting the image into the dimension of shape (1, 30, 30, 3). This is because to predict the traffic sign we have to provide the same dimension we have used when building the model. Then we predict the class, the np.argmax(model.predict(x) axis=-1) returns us a number between (0-42) which represents the class it belongs to. We use the dictionary to get the information about the class.

### Team

1. Saurabh Patil	 
2. Onkar Dasari		 
3. Ananya Mehrotra

