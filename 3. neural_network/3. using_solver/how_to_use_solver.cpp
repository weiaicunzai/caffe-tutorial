#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>
#include <caffe/data_transformer.hpp>
#include <caffe/layers/argmax_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>
#include <google/protobuf/text_format.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace caffe;
using namespace cv;

typedef float Dtype;



/* ========================================= 알아낸 것들 ========================================= */

// 1. NetParameter를 이용해서 Net을 만들면 phase 설정이 안된다. 그냥 트레이닝 시켜버리면 알아서 TRAIN으로?

// 2. Train, Test NetParameter 를 얻는 방법
// p_solver.train_net_param()
// p_solver.test_net_param()


// 3. Test를 위한 Net을 반환해준다는건가?
//solver.test_nets();

// 4. caffe에서 뜬금없이 string이나 char * 가 반환된다면
// 그건 prototxt 의 경로이다. 


// 4. 이건 뭐할 때 쓰는거지
// solver.InitTestNets();
/* =============================================================================================== */



int main()
{
	int iter = 10000;
	Dtype learning_rate = 0.01;

	/* Set mode GPU */
	Caffe::set_mode(Caffe::GPU);


	/* Create Solver Parameter*/
	SolverParameter solver_param;
	ReadSolverParamsFromTextFileOrDie("D:/Github/caffe-tutorial/3. neural_network/3. using_solver/neural_net_solver.prototxt", &solver_param);
	
	/* Create Solver */
	SGDSolver<Dtype> solver(solver_param);

	/* Training */
	solver.Solve();
	
	/* Make NetParameter */
	NetParameter test_net_param;
	ReadNetParamsFromTextFileOrDie("D:/Github/caffe-tutorial/3. neural_network/3. using_solver/neural_net_test.prototxt", &test_net_param);
	
	
	/* Make Net using NetParameter */
	boost::shared_ptr<Net<Dtype>> test_net;
	test_net.reset(new Net<Dtype>(test_net_param));

	/* Get input blob to insert image */
	vector<Blob<Dtype> *> input_blob = test_net->input_blobs();

	/* Get Image and change size */
	Mat mat_img = imread("D:/NewTracking2/bin/mnist/data/test_img/2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat size_changed;
	resize(mat_img, size_changed, cv::Size(input_blob[0]->width(), input_blob[0]->height()));

	/* Convert Mat to Blob */
	TransformationParameter trans_param;
	DataTransformer<Dtype> data_transformer(trans_param, TEST);
	data_transformer.Transform(size_changed, input_blob[0]);

	/* Apply trained layer */
	test_net->CopyTrainedLayersFrom("D:/Github/caffe-tutorial/3. neural_network/3. using_solver/mnist_example_iter_10000.caffemodel");
	
	/* Forward Propagation */
	test_net->Forward();

	/* Get result */
	cout << "Prediction : " << test_net->output_blobs()[0]->cpu_data()[0] << endl;
}