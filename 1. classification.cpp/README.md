# classification.cpp
This file shows how to do classification in caffe. It is implemented by author and I chang a little bit(delete everythoing about mean file cause we don't need mean file now) and add comments.
I added comments as detail as possible, so I expect everybody can understand what's going on in the code easily.
To start with this example file, you need to build caffe. After success to build caffe, you can use this example file
as a main function instead of original one.

## How to run
This program classifies digits from __0 to 9(MNIST)__ using __LeNet__. You need to give network model file, weights, labels and test image. Everything you need is included in this repository. The following instruction is how to run this program.

1. 
