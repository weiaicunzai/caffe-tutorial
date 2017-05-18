# classification.cpp
This file shows how to do classification in caffe. It is implemented by author and I chang a little bit(delete everything about mean file cause we don't need mean file now) and add comments.
I added comments as detail as possible, so I expect everybody can understand what's going on in the code easily.
To start with this example file, you need to build caffe. After success to build caffe, you can use this example file
as a main function instead of original one.

## How to run
This program classifies digits from __0 to 9(MNIST)__ using __LeNet__. You need to give network model file, weights, labels and test image. Everything you need is included in this repository. The following instruction is how to run this program.

1. Build this example file with caffe and make exe file. You might have to remove original main function in caffe.cpp

2. Open cmd and move to the directory which contains that exe file. Suppose that exe file name is caffe.exe

3. Command is `caffe "MODEL_FILE_PATH" "WEIGHT_FILE_PATH" "LABEL_FILE_PATH" "TEST_IMG_PATH"`. Model file is __model/lenet.prototxt__, weight file is __weight/mnist_iter_105000.caffemodel__, label file is __label/label.txt__ and test images are in __test_img__.

4. If you are following perfectly, then you will get classification with great performance.
