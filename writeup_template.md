#**Behavioral Cloning** 

##Writeup Template
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* data.py which loads and preprocess images
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.  All imaging loading and imaging preprocessing is done in the data.py file.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model I am using is detailed in a paper by Nvidia, which can be seen [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The model has about 27 million connections and 250,000 parameters.  This took roughly 6 minutes to train on a data set of 22050 images.  There are three convolutional layers with a filter size of 5x5 and a stride of 2.  The last two convolutional layers have a filter size of 3x3 and a stride of 1.  Lastly the outputs are flattened and there are 5 fully connected layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.  The Nvidia paper did not discuss any use of dropout, but I decided to use it in my model. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.  This was done automatically in keras with the validation_split keyword in model.fit() function. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.  At about 15 mph the vehicle seems to do very well.  Even at 30 mph the vehicle never leaves the track, but some slight swerving can be seen at times.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I used the training data that was given to me from Udacity.  I decided to remove all instances of data points that contained a steering angle of 0.  This really helped my model learn how to deal with sharp turns in the track.  The reason I decided to do this was because the Nvidia paper stated the following: "To remove a bias towards driving straight the training data includes a higher proportion of frames that represent road curves."

I also doubled the size of the training data by flipping all images over the y-axis and multiplying the steering angle by -1.  To speed up training I decided to crop the upper part of the image.  Realistically, the car should not need the trees and the landscape to steer.

Images were also converted from RGB to YUV and were normalized.

###Model Architecture and Training Strategy

####Solution Design Approach

The overall strategy for deriving a model architecture was to start with LeNet and try to see what would work well from there.  LeNet did not perform well on sharp turns, so I decided to use Nvidia's model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the valdiation set.  For the validation set, 20 % of the training set was used.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle was still having troubles with the sharp turns after the bridge.  After going through Nvidia's paper again, I realized that the training set contained a very large amount of data points with 0 steering angle.  This meant that the model was not going to perform well on sharp turns, so I decided to remove these points.

After that, the vehicle was able to deal with sharp turns, but would occasionally get close to the edges.  After augmenting the data set with the left and right camera views and adding a correction value, the vehicle was able to drive autonomously around the track without leaving the road.