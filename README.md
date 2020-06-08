# Convolutional-XGBOOST

This Kernal compares the classification capability of FCN Layers in CNN and XGBoost and Proposing a new model of Convolutional XGBoost which uses Convolutional layers for feature extraction and XGBoost for classification.

We Know that XGBoost is one of the best ensemble learning algorithms which works well for various ML applications. But they don't perform accurately when it comes to Image Classification or other deep learning application. While When we analyse the working of CNN we understand that convolution layers are used to select and extract features for the classification.

## Algorithm
![](https://i.ibb.co/4Th58KC/bg3.png)

As shown in Figure given above, our framework consists of three parts: data preprocessing, feature extraction, and regression analysis. 

In the data processing phase, we divide the raw meta-data features into two parts, i.e., the time-related ones, and the time-unrelated ones. The time-related ones include the “postdate” and “timezone”. The rest are time-unrelated ones, i.e., related to users and items. For the time-related ones, we deal with the “postdate” from the perspective of multiple time scales, and add the “timezone” as an additional feature. Furthermore, we propose a hybrid model 
for the subsequent two phases. More specifically, we use a CNN model consisting of four convolution layers and three full-linear layers on social cues to learn a high-level representation of the data in the feature extraction phase. In the regression analysis phase, we use XGBoost directly to make popularity predictions given the high-level features extracted by CNN.

## Lets Model our CNN

I used the Keras Sequential API, where you have just to add one layer at a time, starting from the input.

The first is the convolutional (Conv2D) layer. It is like a set of learnable filters. I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the two last ones. Each filter transforms a part of the image (defined by the kernel size) using the kernel filter. The kernel filter matrix is applied on the whole image. Filters can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images (feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal value. These are used to reduce computational cost, and to some extent also reduce overfitting. We have to choose the pooling size (i.e the area size pooled each time) more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the net outputs distribution of probability of each class.

## Create XGB model and training it for intermediate values

XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks. However, when it comes to small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now. Please see the chart below for the evolution of tree-based algorithms over the years.
![](https://miro.medium.com/max/1400/1*U72CpSTnJ-XTjCisJqCqLg.jpeg)

### Algorithmic Enhancements:

**Regularization:** It penalizes more complex models through both LASSO (L1) and Ridge (L2) regularization to prevent overfitting.

**Sparsity Awareness:** XGBoost naturally admits sparse features for inputs by automatically ‘learning’ best missing value depending on training loss and handles different types of sparsity patterns in the data more efficiently.

**Weighted Quantile Sketch:** XGBoost employs the distributed weighted Quantile Sketch algorithm to effectively find the optimal split points among weighted datasets.

**Cross-validation:** The algorithm comes with built-in cross-validation method at each iteration, taking away the need to explicitly program this search and to specify the exact number of boosting iterations required in a single run.

## Observation
When I implimented both there is nearly 5-10% increase in accuracy.

### Refrence
Li, Liuwu & Situ, Runwei & Gao, Junyan & Yang, Zhenguo & Liu, Wenyin. (2017). A Hybrid Model Combining Convolutional Neural Network with XGBoost for Predicting Social Media Popularity. 1912-1917. 10.1145/3123266.3127902. 
