#include <iostream>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/layers/accuracy_layer.hpp"


using namespace std;
using namespace caffe;

typedef double Dtype;

/* ========================== Prefix ========================== */
//								//
// Prefix shows what this variable is briefly and clearly	//
// and follwing after prefix explains more details.		//
//								//
// 1. p		=> parameter					//
// 2. b		=> blob						//
// 3. v		=> vector					//
//								//
// Ex.	p_data_layer	=>	Parameter for DataLayer		//
//	b_data_layer	=>	Blob for DataLayer		//
//	v_b_data_layer	=> Vector consistis of b_data_layer	//
//								//
/* ============================================================ */

int main()
{
	Caffe::set_mode(Caffe::GPU);

	int trn_batch_size = 128;
	int tst_batch_size = 10000;
	int class_num = 10;
	int iter = 5000;
	double learning_rate = 0.01;

	/* ========================== Creating Layer ========================== */
	//									//
	// Creating layers always goes thorugh three steps			//
	// a. Make bottom and top blob vectors.					//
	// b. Set LayerParameter which is necessary to create layer		//
	// c. Create layer using LayerParameter	and connect bottom and top.	//
	//									//
	// Then, the question is that how to know what params should be set	//
	// http://caffe.berkeleyvision.org/tutorial/layers.html			//
	// You can check in this website. Click layer you want to know.		//
	// In Parameters section, you can find required parameters		//
	//									//
	/* ==================================================================== */


	/* 1. DataLayer */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_data_layer_bottom;
	vector<Blob<Dtype>*> v_b_data_layer_top;
	
	Blob<Dtype> * b_data = new Blob<Dtype>();
	Blob<Dtype> * b_label = new Blob<Dtype>();

	// DataLayer doesn't have bottom blobs and new two blobs in top.
	v_b_data_layer_top.push_back(b_data);
	v_b_data_layer_top.push_back(b_label);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_data_layer;

	DataParameter * p_data = p_data_layer.mutable_data_param();
	p_data->set_source("D:/NewTracking2/bin/mnist/data/mnist_train_lmdb");
	p_data->set_batch_size(trn_batch_size);
	p_data->set_backend(DataParameter_DB_LMDB);

	TransformationParameter * p_transform = p_data_layer.mutable_transform_param();
	p_transform->set_scale(1. / 255.);

	/* c. Create layer using LayerParamete and connect bottom and top.*/
	DataLayer<Dtype> data_layer(p_data_layer);
	data_layer.SetUp(v_b_data_layer_bottom, v_b_data_layer_top);

	/* 2. InnerProductLayer */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_ip_layer_bottom;
	vector<Blob<Dtype>*> v_b_ip_layer_top;
	
	Blob<Dtype> * const b_ip_layer_top = new Blob<Dtype>();
	
	v_b_ip_layer_bottom.push_back(b_data);
	v_b_ip_layer_top.push_back(b_ip_layer_top);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_ip_layer;
	
	// If you want change some values, you need to use function started with mutable_.
	// See the line below. You can change values with mutable_inner_product_param(),
	// but you can't with inner_product_param() function because it returns const.
	InnerProductParameter * p_ip = p_ip_layer.mutable_inner_product_param();
	p_ip->set_num_output(10);
	
	FillerParameter * p_weight_filler = p_ip->mutable_weight_filler();
	p_weight_filler->set_type("xavier");
	
	FillerParameter * p_bias_filler = p_ip->mutable_bias_filler();
	p_bias_filler->set_type("constant");

	/* c. Create layer using LayerParameter and connect bottom and top. */
	InnerProductLayer<Dtype> ip_layer(p_ip_layer);
	ip_layer.SetUp(v_b_ip_layer_bottom, v_b_ip_layer_top);

	/* 3. Sigmoid & Loss layer */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_loss_layer_top;
	vector<Blob<Dtype>*> v_b_loss_layer_bottom;

	Blob<Dtype> * const b_loss_layer_top = new Blob<Dtype>();
	v_b_loss_layer_top.push_back(b_loss_layer_top);
	v_b_loss_layer_bottom.push_back(b_ip_layer_top);
	v_b_loss_layer_bottom.push_back(b_label);

	int n = b_ip_layer_top->num();
	int c = b_ip_layer_top->channels();
	int w = b_ip_layer_top->width();
	int h = b_ip_layer_top->height();
	cout << "N = " << n << ", C = " << c << ", W = " << w << ", H = " << h << endl;

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_loss_layer;
	
	//SigmoidCrossEntropyLossLayer<Dtype> loss_layer(p_loss_layer);
	SoftmaxWithLossLayer<Dtype> loss_layer(p_loss_layer);
	loss_layer.SetUp(v_b_loss_layer_bottom, v_b_loss_layer_top);


	/* ============================== Training ============================ */
	//									//
	// Traing always goes thorugh three steps and repeats thes three steps. //
	// a. Forward Propagation.						//
	// b. Back Propagation.							//
	// c. Apply learning rate.						//
	// d. Update weights.							//
	//									//
	/* ==================================================================== */

	for (int i = 0; i < 5000; i++)
	{
		/* a. Forward Propagation. */		
		data_layer.Forward(v_b_data_layer_bottom, v_b_data_layer_top);
		ip_layer.Forward(v_b_ip_layer_bottom, v_b_ip_layer_top);
		Dtype loss = loss_layer.Forward(v_b_loss_layer_bottom, v_b_loss_layer_top);
		
		if(i%100 == 0)
			cout << "Iter : " << i << " / Loss : " << loss << endl;

		/* b. Back Propagation. */
		// prop_down selects blobs for back propagation in case that bottom blob vector has multiple blobs.
		// loss_layer has two bottom blobs, the top blob of ip_layer and b_label.
		// Only the top blob of ip_layer is connected with data_layer, 
		// so I gave true to this blob and false to b_label.
		vector<bool> prop_down;
		prop_down.push_back(1);
		prop_down.push_back(0);

		loss_layer.Backward(v_b_loss_layer_top, prop_down, v_b_loss_layer_bottom);
		ip_layer.Backward(v_b_ip_layer_top, prop_down, v_b_ip_layer_bottom);
		// data_layer doesn't have bottom blob.

		/* c. Apply learning rate. */
		vector<caffe::shared_ptr<Blob<Dtype>>> learnable_blob = ip_layer.blobs();
		caffe_gpu_scale(learnable_blob[0]->count(), learning_rate, learnable_blob[0]->gpu_diff(), learnable_blob[0]->mutable_gpu_diff());

		/* d. Update weights. */
		learnable_blob[0]->Update();
	}

	/* ================================ Test ============================== */
	//									//
	// a. Change DataParameter to get test data instead of training data.	//
	// b. Reshape InnerProductLayer	cause batch size is changed.		//
	// c. Remove SoftmaxWitlLosslayer and attach AccuracyLayer.		//
	// d. Forward Propagation and get accuracy.				//
	//									//
	/* ==================================================================== */

	/* a. Change DataParameter to get test data instead of training data. */
	// Don't need to make DataLayer again.
	// Just change the parameter values. 
	// LayerParameter using this parameter is already made,
	// and DataLayer using that LayerParameter is also made.
	tst_batch_size = 10000;
	p_data->set_source("D:/NewTracking2/bin/mnist/data/mnist_test_lmdb");
	p_data->set_batch_size(tst_batch_size);

	/* b. Reshape InnerProductLayer cause batch size is changed. */
	ip_layer.Reshape(v_b_ip_layer_bottom, v_b_ip_layer_top);

	/* c. Remove SoftmaxWitlLosslayer and attach AccuracyLayer. */
	vector<Blob<Dtype> *> v_b_accuracy_layer_top;
	vector<Blob<Dtype> *> v_b_accuracy_layer_bottom;

	// Accuracy Layer needs two bottom blobs, result of ip_layer(prediction) and b_label(answer or groud truth)
	Blob<Dtype> * b_accuracy_layer_top = new Blob<Dtype>();
	v_b_accuracy_layer_top.push_back(b_accuracy_layer_top);
	v_b_accuracy_layer_bottom.push_back(b_ip_layer_top);
	v_b_accuracy_layer_bottom.push_back(b_label);

	LayerParameter p_accuracy_layer;
	AccuracyParameter * p_accuracy = p_accuracy_layer.mutable_accuracy_param();
	AccuracyLayer<Dtype> accuracy_layer(p_accuracy_layer);
	accuracy_layer.SetUp(v_b_accuracy_layer_bottom, v_b_accuracy_layer_top);

	// d. Forward Propagation and get accuracy.
	data_layer.Forward(v_b_data_layer_bottom, v_b_data_layer_top);
	ip_layer.Forward(v_b_ip_layer_bottom, v_b_ip_layer_top);
	accuracy_layer.Forward(v_b_accuracy_layer_bottom, v_b_accuracy_layer_top);
	
	const Dtype * accuracy = b_accuracy_layer_top->cpu_data();

	cout << "Accuracy : " << accuracy[0] * 100 << "%" << endl;
}
