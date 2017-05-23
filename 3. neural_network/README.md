# Neural Network
Implementation of the basic neural network which has two fully connected layers. There are two parts of this example.

## 1. without_prototxt
caffe is using prototxt file as a model defintion. Many of people may have seen model definition in prototxt file but few people may
have seen what's going on really in caffe something like how to make layers, how to sum up layers to build big networks, how to train
and so on just using c++ and caffe. This example will let you know how to do it.

## 2. using_prototxt
In this example, make networks, train and get accuracy using simple neural network prototxt file. This example shows how to make
network using prototxt file and manage it. You will see how to make network from prototxt file, pick some layers up in the network 
and manage them to make your own machine.

## Finally...
After compare these two methods, you will catch that using prototxt is much more convenient because you need to define whole model
in long c++ code without prototxt.
