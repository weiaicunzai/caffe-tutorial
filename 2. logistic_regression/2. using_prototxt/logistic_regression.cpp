#include <caffe/caffe.hpp>

#include <string>
#include <vector>


using namespace caffe;
using namespace std;

typedef double Dtype;

int main()
{
	int iter = 5000;
	Dtype learning_rate = 0.1;

	/* Set mode GPU */
	Caffe::set_mode(Caffe::GPU);

	/* Create Network */
	const string model_file = "D:/Github/caffe-tutorial/2. logistic_regression/2. using_prototxt/logistic_regression.prototxt";
	boost::shared_ptr<Net<Dtype>> network;
	network.reset(new Net<Dtype>(model_file, TRAIN));
	
	/* Training */
	for (int i = 0; i < iter; i++)
	{
		/* Forward Propagation */
		network->Forward();
		vector<Blob<Dtype>*> loss = network->output_blobs();

		/* Get loss */
		if(i%100 == 0)
			cout << "Loss : " << loss[0]->cpu_data()[0] << endl;

		/* Back Propagation */
		network->Backward();

		/* Apply learning rate */
		// Only InnerProductLayer can update weights. So, get InnerProductLayer.
		boost::shared_ptr<Layer<Dtype>> learnable_layer =  network->layer_by_name("fully_connected");

		// Get weight blob of InnerProductLayer
		vector<boost::shared_ptr<Blob<Dtype>>> learnable_blob = learnable_layer->blobs();

		// Scale gradients with learning rate
		caffe_scal(learnable_blob[0]->count(), learning_rate, learnable_blob[0]->mutable_cpu_diff());

		/* Another implementation of learnable_layer and learnable_blob */
		//vector<boost::shared_ptr<Layer<Dtype>>> all_layers = network->layers();
		//boost::shared_ptr<Layer<Dtype>> learnable_layer = all_layers[1];
		//vector<boost::shared_ptr<Blob<Dtype>>> learnable_blob = learnable_layer->blobs();

		/* Update */
		network->Update();
	}

	/* Create network for test*/
	boost::shared_ptr<Net<Dtype>> network_;

	// This is new network, so weights are initialized randomly.
	network_.reset(new Net<Dtype>(model_file, TEST));
	
	// Get trained weights
	network_->ShareTrainedLayersWith(network.get());

	/* Forward Propagation and get accuracy*/
	network_->Forward();
	vector<Blob<Dtype>*> accuracy = network_->output_blobs();
	cout << accuracy[0]->cpu_data()[0] * 100 << "%" << endl;

}