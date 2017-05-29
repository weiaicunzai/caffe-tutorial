#include <iostream>
#include <vector>

#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <caffe/blob.hpp>

#include <caffe/layers/data_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/sigmoid_layer.hpp>
#include <caffe/layers/softmax_loss_layer.hpp>
#include <caffe/layers/accuracy_layer.hpp>

typedef double Dtype;

using namespace std;
using namespace caffe;

/* ======================== prefix ======================== */
//															//
// prefix shows what this variable is briefly and clearly	//
// and follwing after prefix explains more details.			//
//															//
// 1. p		=> parameter									//
// 2. b		=> blob											//
// 3. v		=> vector										//
//															//
// ex. p_data_layer	=>	parameter for datalayer				//
//	   b_fc1_top	=>	top blob for fc1					//
//	   v_b_fc1_top	=>	vector consistis of b_fc1_top		//
//															//
/* ======================================================== */

int main()
{
	int iter = 30000;
	Dtype learning_rate = 0.1;

	int trn_batch = 128;
	int tst_batch = 10000;

	int fc1_output_num = 128;
	int fc2_output_num = 10;

	string trn_data_path = "D:/NewTracking2/bin/mnist/data/mnist_train_lmdb";
	string tst_data_path = "D:/NewTracking2/bin/mnist/data/mnist_test_lmdb";

	Caffe::set_mode(Caffe::GPU);
	
	/* ========================= Creating Layer ========================= */
	//																	  //
	// Creating layers always goes thorugh three steps					  //
	// a. Make bottom and top blob vectors.								  //
	// b. Set LayerParameter which is necessary to create layer			  //
	// c. Create layer using LayerParameter	and connect bottom and top.	  //
	//																	  //
	// Then, the question is that how to know what params should be set	  //
	// http://caffe.berkeleyvision.org/tutorial/layers.html				  //
	// You can check in this website. Click layer you want to know.		  //
	// In Parameters section, you can find required parameters			  //
	//																	  //
	/* ================================================================== */

	/* ========================= 1. DataLayer ========================= */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_data_bot;
	vector<Blob<Dtype>*> v_b_data_top;
	
	Blob<Dtype> * b_data = new Blob<Dtype>();
	Blob<Dtype> * b_label = new Blob<Dtype>();
	
	// DataLayer doesn't have bottom blobs and new two blobs in top.
	v_b_data_top.push_back(b_data);
	v_b_data_top.push_back(b_label);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_data_layer;
	DataParameter * p_data = p_data_layer.mutable_data_param();
	p_data->set_source(trn_data_path);
	p_data->set_backend(DataParameter_DB_LMDB);
	p_data->set_batch_size(trn_batch);
	TransformationParameter * p_transform = p_data_layer.mutable_transform_param();
	p_transform->set_scale(1. / 255.);

	/* c. Create layer using LayerParamete and connect bottom and top.*/
	DataLayer<Dtype> data_layer(p_data_layer);
	data_layer.SetUp(v_b_data_bot, v_b_data_top);

	/* ========================= 2. InnerProductLayer_1 ========================= */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_fc1_bot;
	vector<Blob<Dtype>*> v_b_fc1_top;
	
	Blob<Dtype> * b_fc1_top = new Blob<Dtype>();
	
	v_b_fc1_bot.push_back(b_data);	// 784
	v_b_fc1_top.push_back(b_fc1_top);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_fc1_layer;
	
	// If you want change some values, you need to use function started with mutable_.
	// See the line below. You can change values with mutable_inner_product_param(),
	// but you can't with inner_product_param() function because it returns const.
	InnerProductParameter * p_fc1 = p_fc1_layer.mutable_inner_product_param();
	p_fc1->set_num_output(fc1_output_num);	//256

	FillerParameter * p_fc1_weight = p_fc1->mutable_weight_filler();
	p_fc1_weight->set_type("xavier");
	
	FillerParameter * p_fc1_bias = p_fc1->mutable_bias_filler();
	p_fc1_bias->set_type("constant");

	/* c. Create layer using LayerParameter and connect bottom and top. */
	InnerProductLayer<Dtype> fc1_layer(p_fc1_layer);
	fc1_layer.SetUp(v_b_fc1_bot, v_b_fc1_top);

	/* ========================= 3. SigmoidLayer_1 ========================= */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_sig_bot;
	vector<Blob<Dtype>*> v_b_sig_top;
	
	Blob<Dtype> * b_sig_top = new Blob<Dtype>();

	v_b_sig_bot.push_back(b_fc1_top);
	v_b_sig_top.push_back(b_sig_top);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_sig_layer;
	SigmoidParameter * p_sig = p_sig_layer.mutable_sigmoid_param();
	
	/* c. Create layer using LayerParameter and connect bottom and top. */
	SigmoidLayer<Dtype> sig_layer(p_sig_layer);
	sig_layer.SetUp(v_b_sig_bot, v_b_sig_top);

	/* ========================= 4. InnerProductLayer_2 ========================= */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_fc2_bot;
	vector<Blob<Dtype>*> v_b_fc2_top;

	Blob<Dtype> * b_fc2_top = new Blob<Dtype>();

	v_b_fc2_bot.push_back(b_sig_top);
	v_b_fc2_top.push_back(b_fc2_top);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_fc2_layer;

	InnerProductParameter * p_fc2 = p_fc2_layer.mutable_inner_product_param();
	p_fc2->set_num_output(fc2_output_num);

	FillerParameter * p_fc2_weight = p_fc2->mutable_weight_filler();
	p_fc2_weight->set_type("xavier");

	FillerParameter * p_fc2_bias = p_fc2->mutable_bias_filler();
	p_fc2_bias->set_type("constant");

	/* c. Create layer using LayerParameter and connect bottom and top. */
	InnerProductLayer<Dtype> fc2_layer(p_fc2_layer);
	fc2_layer.SetUp(v_b_fc2_bot, v_b_fc2_top);


	/* ========================= 5. Softmax & Loss layer ========================= */
	/* a.  Make bottom and top blob vectors. */
	vector<Blob<Dtype>*> v_b_loss_bot;
	vector<Blob<Dtype>*> v_b_loss_top;
	Blob<Dtype> * b_loss_top = new Blob<Dtype>();

	// SoftmaxWithLossLayer needs two bottom blobs.
	v_b_loss_bot.push_back(b_fc2_top);
	v_b_loss_bot.push_back(b_label);
	v_b_loss_top.push_back(b_loss_top);

	/* b. Set LayerParameter which is necessary to create layer. */
	LayerParameter p_loss_layer;

	/* c. Create layer using LayerParameter and connect bottom and top. */
	SoftmaxWithLossLayer<Dtype> loss_layer(p_loss_layer);
	loss_layer.SetUp(v_b_loss_bot, v_b_loss_top);

	/* ============================== Training ============================ */
	//																		//
	// Traing always goes thorugh three steps and repeats these	three steps.//
	// a. Forward Propagation.												//
	// b. Back Propagation.													//
	// c. Apply learning rate.												//
	// d. Update weights.													//
	//																		//
	/* ==================================================================== */
	for (int i = 0; i < iter; i++)
	{
		/* a. Forward Propagation */
		data_layer.Forward(v_b_data_bot, v_b_data_top);
		fc1_layer.Forward(v_b_fc1_bot, v_b_fc1_top);
		sig_layer.Forward(v_b_sig_bot, v_b_sig_top);
		fc2_layer.Forward(v_b_fc2_bot, v_b_fc2_top);
		Dtype loss = loss_layer.Forward(v_b_loss_bot, v_b_loss_top);

		if (i % 1000 == 0)
			cout << "Iter : " << i+100 << " / Loss : " << loss << endl;

		/* b. Back Propagation */
		// prop_down selects blobs for back propagation in case that bottom blob vector has multiple blobs.
		// loss_layer has two bottom blobs, the top blob of fc2_layer and b_label.
		// Only the top blob of fc2_layer is connected with sig_layer, 
		// so I gave true to this blob and false to b_label.
		vector<bool> prop_down;
		prop_down.push_back(1);
		prop_down.push_back(0);

		loss_layer.Backward(v_b_loss_top, prop_down, v_b_loss_bot);
		fc2_layer.Backward(v_b_fc2_top, prop_down, v_b_fc2_bot);
		sig_layer.Backward(v_b_sig_top, prop_down, v_b_sig_bot);
		fc1_layer.Backward(v_b_fc1_top, prop_down, v_b_fc1_bot);

		/*c. Apply learning rate. & d. Update weights. */
		vector<boost::shared_ptr<Blob<Dtype>>> fc2_learnable_blob = fc2_layer.blobs();
		caffe_scal(fc2_learnable_blob[0]->count(), learning_rate, fc2_learnable_blob[0]->mutable_cpu_diff());
		fc2_learnable_blob[0]->Update();

		vector<boost::shared_ptr<Blob<Dtype>>> fc1_learnable_blob = fc1_layer.blobs();
		caffe_scal(fc1_learnable_blob[0]->count(), learning_rate, fc1_learnable_blob[0]->mutable_cpu_diff());
		fc1_learnable_blob[0]->Update();
	}

	/* ================================ Test ============================== */
	//																		//
	// a. Change DataParameter to get test data instead of training data.	//
	// b. Reshape InnerProductLayer	cause batch size is changed.			//
	// c. Remove SoftmaxWitlLosslayer and attach AccuracyLayer.				//
	// d. Forward Propagation and get accuracy.								//
	//																		//
	/* ==================================================================== */

	/* a. Change DataParameter to get test data instead of training data. */
	p_data->set_source(tst_data_path);
	p_data->set_batch_size(tst_batch);

	/* b. Reshape InnerProductLayer cause batch size is changed. */
	fc1_layer.Reshape(v_b_fc1_bot, v_b_fc1_top);
	fc2_layer.Reshape(v_b_fc2_bot, v_b_fc2_top);	// fc2의 input은 fc1의 output이라 변화 없을듯

	/* c. Remove SoftmaxWitlLosslayer and attach AccuracyLayer. */
	vector<Blob<Dtype>*> v_b_accr_bot;
	vector<Blob<Dtype>*> v_b_accr_top;

	Blob<Dtype>* b_accr_top = new Blob<Dtype>();

	v_b_accr_bot.push_back(b_fc2_top);
	v_b_accr_bot.push_back(b_label);
	v_b_accr_top.push_back(b_accr_top);

	LayerParameter p_accr_layer;
	AccuracyLayer<Dtype> accr_layer(p_accr_layer);
	accr_layer.SetUp(v_b_accr_bot, v_b_accr_top);

	/* d. Forward Propagation and get accuracy. */
	data_layer.Forward(v_b_data_bot, v_b_data_top);
	fc1_layer.Forward(v_b_fc1_bot, v_b_fc1_top);
	sig_layer.Forward(v_b_sig_bot, v_b_sig_top);
	fc2_layer.Forward(v_b_fc2_bot, v_b_fc2_top);
	accr_layer.Forward(v_b_accr_bot, v_b_accr_top);
	
	const Dtype * accuracy = b_accr_top->cpu_data();

	cout << "Accuracy : " << accuracy[0]*100 << "%" << endl;
}