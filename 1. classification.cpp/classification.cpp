#define USE_OPENCV  // I just added

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;




// ################################ CLASS DEFINITION ################################ //

class Classifier {
public:
	Classifier(const string& model_file,	// deploy.prototxt		=> This file describe network model.
		const string& trained_file,			// network.caffemodel	=> Weights of the network model.
		const string& label_file);			// labels.txx			=> Class label

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;		// Representing the network, from input to output
	cv::Size input_geometry_;			// Width and height of input
	int num_channels_;					// Number of channels
	std::vector<string> labels_;		// Labels
};

// ################################ END OF CLASS DEFINITION ################################ //





// ################################ CONSTRUCTOR ################################ //

Classifier::Classifier(const string& model_file,	// deploy.prototxt		=> This file describe network model.
	const string& trained_file,						// network.caffemodel	=> Weights of the network model.
	const string& label_file) {						// labels.txt			=> Class label
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);	// Use CPU
#else
	Caffe::set_mode(Caffe::GPU);	// Use GPU
#endif
	/* Load the network. */
	// To understand this line, you need to know 'shared_ptr' in c+11.
	// Easily speaking, net_ is a pointer looking at Net class object. But it is a shared pointer,
	// it is free from memory management cause shared pointer is doing it automatically.
	// So, it is a dynamic allocation without danger of memory leak.

	// reset is function which is implemented in memory (see the header files above)
	// Make a new Net class object By 'new Net<float>(model_file, TEST)'
	// This object has network defined in 'model_file' and phase is TEST (There are two phases, TRAIN and TEST)
	// and pass it to the shared_ptr net_ by reset function.
	net_.reset(new Net<float>(model_file, TEST));




	// Appy network.caffemodel(Weights) to the net_.
	net_->CopyTrainedLayersFrom(trained_file);




	// num_inputs() returns net_input_blobs_.size();
	// num_outputs() returns net_output_blobs_.size();
	// Explanation about blob is following below.
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";




	// Blob is a key data structure in caffe.
	// It represents such as number of data, number of channels, width and height
	// In other words, it represents layer shape or data.

	// input_blobs() function returns vector<Blob<Dtype>*> net_input_blobs_;
	// If there were no problem in two CHECK_EQs above, then net_input_blobs_ is a vector
	// with size of one.

	// net_->input_blobs()[0] is Blob<Dtype> *, so input_layer represents a blob of input layer
	// In other words, input_layer represents all about input layer
	Blob<float>* input_layer = net_->input_blobs()[0];



	// Set channel number
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";



	// Set data size (width and height)
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());



	/* Load labels. */
	// labels_ is vector<string>.
	// Push back each line to the labels_ from label file.
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));



	// Same with Blob<float>* input_layer = net_->input_blobs()[0];
	// output_layer is blob of output layer
	Blob<float>* output_layer = net_->output_blobs()[0];




	// output_layer->channel() is the number of classes
	// output layer blob => Number=1 / Width=1 / Height=1 / Channel=number of classes
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

// ################################ END OF CONSTRUCTOR ################################ //





// ################################ UTIL FUNCTIONS ################################ //

// lhs and rhs is pair of float and int
// This function returns true if lhs float data > rhs float data
// It will be used in classification confidence sorting
static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}




/* Return the indices of the top N values of vector v. */
// vector<float> v is a classification confidence or probability vector.
// Argmax function find top N confidence values and return indices of them.
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;

	// Make (confidence, index) pair.
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));

	// Sort with confidence values.
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;

	// Get indices of top N confidence values. 
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

// ################################ END OF UTIL FUNCTIONS ################################ //





// ################################ CLASS MEMBER FUNCTIONS ################################ //

/* =================================================================
* Name		:	Classify
* Parameter :	img => Input image
				N => N of top N predictions
* Action	:	Returns top N predictions for input image
==================================================================== */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {

	// Do forward propagation and get prediction
	// output[i] is confidence that img belongs to class i
	std::vector<float> output = Predict(img);




	// std::min<int> returns smaller one between labels_.size() and N.
	// labels_.size() is whole classes number and N is parameter you decided
	N = std::min<int>(labels_.size(), N);




	// Get top N indices of output
	std::vector<int> maxN = Argmax(output, N);




	// Save the prediction result(Prediction : pair of confidence and index) here
	std::vector<Prediction> predictions;




	// Insert results to vector<Prediction> predictions
	// For example labels_ = {apple, orange, melon} / output = {0.7, 0.2, 0,1}
	// Then classification of input image is ...
	// apple with probability of 0.7
	// orange with probability of 0.2
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}


/* =================================================================
* Name		:	Predict
* Parameter :	img => Input image
				N	=> N of top N predictions
* Action	:	Returns confidence vector for each class
==================================================================== */
std::vector<float> Classifier::Predict(const cv::Mat& img) {

	// Get input layer blob.
	Blob<float>* input_layer = net_->input_blobs()[0];



	// Reshape input layer blob. Only one change is the number of data. It is changed to 1.
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);



	/* Forward dimension change to all layers. */
	//Input layer blob is changed above, so we need to change all following layers sizes.
	//Reshape() function is doing that. We don't need to do forward propagation to apply this change
	net_->Reshape();



	// An elementry of input_channels vector represents one channel data of input layer
	// After WrapInputLayer, we can modify input layer blob by modifying input_channels
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);



	// Convert the img to the input layer blob size and pass data to input_channels
	// It means that set input layer blob data to img data.
	Preprocess(img, &input_channels);



	// Forward propagation
	net_->Forward();



	// Get output layer blob
	Blob<float>* output_layer = net_->output_blobs()[0];



	// Get first elemetent's address of blob data
	const float* begin = output_layer->cpu_data();



	// output_layer->channels() returns the number of class
	const float* end = begin + output_layer->channels();



	// Make a confidence value vector
	return std::vector<float>(begin, end);
}



/* =================================================================
* Name		:	WrapInputLayer
* Parameter :	input_channels => Wrapper of input layer blob
* Action	:	Make input_channels Wrapper of input layer blob,
				so we can change input layer blob by changing input_channels
==================================================================== */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {

	// Get input layer blob
	Blob<float>* input_layer = net_->input_blobs()[0];



	// Get input layer blob width and height
	int width = input_layer->width();
	int height = input_layer->height();



	// Get input layer blob data and save it to one-dimensional array.
	// Also, by using mutable_cpu_data(), we can change blob data using input_data.
	float* input_data = input_layer->mutable_cpu_data();



	for (int i = 0; i < input_layer->channels(); ++i) {
		
		// Make cv::Mat per a channel.
		// input_data is input layer blob data and 
		// this constructor means that make a cv::Mat that width is width(param), height is height(param),
		// data type is CV_32FC1 and data (exactly starting pointer of data) is input_data.
		// input_data is one dimensional array and if input_data length is longer than height * width
		// then just take from input_data(starting pointer) to (input_data + (height * width) - 1)(ending pointer).
		// So, we got a cv::Mat of only one channel.
		// If the input_data has data of 3 channels, then its length will be width * height * 3
		// and we need to make other two cv::Mat and it is implemented by for(int i=0; i< input_layer->channels(), ++i)
		cv::Mat channel(height, width, CV_32FC1, input_data);



		// Insert this cv::Mat to the input_channels vector(This is a parameter of this function)
		// The important thing is that input_channels is a vector of cv::Mat channel,
		// cv::Mat channel has float * input_data as a pixel data.
		// float * input_data is pointing input layer blob data directly
		// It means that we can change input layer blob data by changing vector<Mat> * input_channels.
		// This is the PURPOSE OF THE WrapInputLayer FUNCTION.
		input_channels->push_back(channel);



		// Move the starting pointer to the next channel.
		input_data += width * height;
	}
}



/* =================================================================
* Name		:	Preprocess
* Parameter :	img				=> Input image
				input_channels	=> Wrapper of input layer blob 
* Action	:	Set input layer blob data to input image data.
				Preprocess for forward propagation.
==================================================================== */
void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	
	
	/* Convert the input image to the input image format of the network. */
	// num_channels_ is the number of channel of input layer.
	// Make img channel same with input_channels_.
	// As a converting result, we get cv::Mat sample.
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;



	// Make the input images size same with input layer blob size
	// As a converting result, we get cv::Mat sample_resized
	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;



	// Make input image data type same with input layer blob data type
	// As a converting result, we get cv::Mat sample_float
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);



	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */

	// Split function splits sample_float into each channel, makes cv::Mat per a channel and
	// saves it to vector<Mat> input_channels.
	// We already initialized input_channels using WrapInputLayer,
	// so saving data in input_channels means saving data in input layer blob

	// The most important thing is that Mat sample_float should have exactly same properties,
	// such as width, height and data type(ex. CV_32FC1),
	// with input_channels->at[idx].data,
	// or you will get error "Input channels are not wrapping the input layer of the network."
	// You can see this error message below.
	cv::split(sample_float, *input_channels);



	// In the WrapInputLayer function, 
	// we made that it is possible to change input layer blob
	// by changing vector<Mat> * input_channels.
	// This statement is checking that.
	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

// ################################ END OF CLASS MEMBER FUNCTIONS ################################ //


int main(int argc, char** argv) {
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel"
			<< " mean.binaryproto labels.txt img.jpg" << std::endl;
		return 1;
	}

	caffe::GlobalInit(&argc, &argv);

	// Get necessary file
	string model_file = argv[1];
	string trained_file = argv[2];
	string label_file = argv[3];
	string file = argv[4];

	// Create classifier
	Classifier classifier(argv[1], argv[2], argv[3]);


	std::cout << "---------- Prediction for "
		<< file << " ----------" << std::endl;

	// Get input image
	cv::Mat img = cv::imread(file, -1);
	CHECK(!img.empty()) << "Unable to decode image " << file;

	// Classify the image and get pair of confidence and class number
	std::vector<Prediction> predictions = classifier.Classify(img);

	/* Print the top N predictions. */
	for (size_t i = 0; i < predictions.size(); ++i) {
		std::cout << std::fixed << std::setprecision(4) << predictions[i].second << " - \""
			<< predictions[i].first << "\"" << std::endl;
	}
}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  