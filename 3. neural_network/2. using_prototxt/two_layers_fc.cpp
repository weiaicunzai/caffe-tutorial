#include <caffe/caffe.hpp>

#include <string>
#include <vector>

using namespace std;
using namespace caffe;

typedef double Dtype;

int main()
{
	int iter = 10000;
	Dtype learning_rate = 0.01;

	/* Set mode GPU */
	Caffe::set_mode(Caffe::GPU);

	/* Create Network */
	string model_file = "D:/Github/caffe-tutorial/3. neural_network/2. using_prototxt/neural_network.prototxt";
	boost::shared_ptr<Net<Dtype>> network_training;
	network_training.reset(new Net<Dtype>(model_file, TRAIN));

	/* Training */
	for (int i = 0; i < iter; i++)
	{
		/* Forward Propagation */
		network_training->Forward();

		/* Get loss */
		vector<Blob<Dtype>*> loss_blob = network_training->output_blobs();
		Dtype loss = loss_blob[0]->cpu_data()[0];
		
		if (i % 100 == 0)
			cout << "Iter : " << i << " / Loss : " << loss << endl;

		/* Back Propagation */
		network_training->Backward();
		boost::shared_ptr<Layer<Dtype>> fc1_layer = network_training->layer_by_name("fc1");
		boost::shared_ptr<Layer<Dtype>> fc2_layer = network_training->layer_by_name("fc2");

		/* Get weight blobs of two InnerProductLayers */
		boost::shared_ptr<Blob<Dtype>> fc1_learnable_blob = fc1_layer->blobs()[0];
		boost::shared_ptr<Blob<Dtype>> fc2_learnable_blob = fc2_layer->blobs()[0];

		/* Apply learning rate */
		caffe_scal(fc1_learnable_blob->count(), learning_rate, fc1_learnable_blob->mutable_cpu_data());
		caffe_scal(fc2_learnable_blob->count(), learning_rate, fc2_learnable_blob->mutable_cpu_data());

		/* Update */
		fc1_learnable_blob->Update();
		fc2_learnable_blob->Update();
	}
	/* Create network for test*/
	boost::shared_ptr<Net<Dtype>> network_test;
	network_test.reset(new Net<Dtype>(model_file, TEST));

	/* Get trained weights */
	network_test->ShareTrainedLayersWith(network_training.get());
	
	/* Forward Propagation and get accuracy*/
	network_test->Forward();
	vector<Blob<Dtype>*> accuracy_blob = network_test->output_blobs();
	cout << "Accuracy : " << accuracy_blob[0]->cpu_data()[0] << endl;
}