Behavioral Cloning
---
This project involves using a convolutional neural network to train a vehicle to drive autonomously by controlling the steering wheel angles.
model.py is used to generate a model that drives a vehicle by predicting steering wheel angles for a given frame.  The model comes from a paper by [Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The actual model is stored in model.h5, which was generated from keras.  The data set that is used in model.py can be found [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) from Udacity.

To learn more about the thought process behind the model and image processing, check out writeup_template.md.

To run: ```python drive.py model.h5```
and then start up the simulator program in autonomous mode.

For Windows the simulator can be found [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip).

For more details about the project, check out Udacity's [description](https://github.com/udacity/CarND-Behavioral-Cloning-P3).