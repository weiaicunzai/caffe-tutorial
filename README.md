# caffe-tutorial
__Caffe tutorial for the beginners__

Caffe is a deep learning framework. This is a tutorial for beginners and self study repo. You need to build caffe (Maybe the most challenging)
to follow this example. I am using __windows 10__ and __visual studio 2013__ but I don't think this is a big deal because the code will not be
changed.

At first, caffe provides __a simple classification example__. So, let's get it started understanding this example. 

Usaually, caffe needs __three additional files__ to make it easy. These are __model definition file__, __solver file__ and __weight file__. Model
defintion file and solver file is __prototxt file__ (ex. model.prototxt, solver.prototxt) and weight file is __caffemodel file__ (ex. mnist_iter_10000.caffemodel).
You can decide whether use these convenient file or not, so I want to show all these cases.

1. Implementation using __solver__ and __model definition file__
2. Implementation using model __definition file__
3. Implementation using __nothing__
