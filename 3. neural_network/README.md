# Neural Network
Implementation of the basic neural network which has two fully connected layers. There are two parts of this example.

## 1. using_solver
caffe is using prototxt file as a solver definition which is really helpful and covenient for training. This example shows how to make
and use solver.

## 2. using_prototxt
In this example, make networks, train and get accuracy using simple logistic regression prototxt file. This example shows how to make network using prototxt file, pick some layers up in the network and manage them to make your own machine.

## 3. without_prototxt
caffe is using prototxt file as a model defintion. Many of people may have seen model definition in prototxt file and pass it to the caffe but few people may have seen what's going on really in caffe something like how to make layers, how to sum up layers to build big networks, how to train and so on. This example will let you know how to do it by implementing everything without prototxt file.

## Finally...
After compare these two methods, you will catch that using prototxt is much more convenient because you need to define whole model
in long c++ code without prototxt.
