// neurol-net-tut.cpp

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::operator<<;
using std::ifstream;
using std::ofstream;
using std::stringstream;
// Silly class to read training data from a file 

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned> &topology);

		// Returns the number of input values read from the file
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);
	
private:
	ifstream m_trainingDataFile;

};

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals) {
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}

void TrainingData::getTopology(vector<unsigned> &topology) {
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() ||
		label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals) {
	inputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}


struct Connection
{
	double weight;
	double deltaweight;
};

class Neuron;

typedef vector<Neuron> Layer;

/**
	*** class Neuron ***
*/

class Neuron {
public:
	// construcor needs to create a vector of connections
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradient(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;

	double m_outputVal;
	vector<Connection> m_outputWeights; // each neuron to the right has a weight
	unsigned m_myIndex;
	double m_gradient;

	static double eta; // [0.0...1.0] overall net learning rate
	static double alpha; // [0.0...n] multiplier of last weight change (momentum)
};

double Neuron::eta = 0.15; // [0.0...1.0] overall net learning rate
double Neuron::alpha = 0.5; // [0.0...n] multiplier of last weight change (momentum)


void Neuron::updateInputWeights(Layer &prevLayer) {

	// the weights to be updated are in the connection constructor
	// in the neurons  int the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight;

		double newDeltaWeight =
			// Individual input magnified by the gradient and train  rate
			eta
			* neuron.getOutputVal()
			* m_gradient
			// Also add momentum = a fraction of the previous delta weight
			+ alpha
			*  oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

	}
}

double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed 
	for ( unsigned n = 0; n < nextLayer.size() - 1; ++n){
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradient(const Layer &nextLayer) {
	// it`s similar to output grad but has a little diffenrens.
	// It`s an actual output that we are expecting to see, which is 
	// uknown at this step
	// sum of der of the next layer

	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal); // FAIL !!! / -> *

}

// see the diff between actual and getting values 
void Neuron::calcOutputGradients(double targetVal) {

	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);

}

double Neuron::transferFunction(double x) {

	// tanh - output range [-1.0 ... 1.0]
	return tanh(x);

}

double Neuron::transferFunctionDerivative(double x) {
	// tanh derivative
	// approximation using x squared
	return 1.0 - x*x;

}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer & prevLayer)
{// math part 
	double sum = 0.0;

	// Sum the previous layer`s outputs (which are out inputs)
	// Inlcude the bias node from the previous layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

/**
		*** class Net ***
*/
class Net {

public:
	Net(vector<unsigned>  &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const ; //  it`s not modify Net
	double getRecentAverageError(void) const { return m_recentAverageError; }
private:
	vector<Layer> m_layers; // m_layers[layer_num][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<double> &resultVals) const {
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}

}

void Net::backProp(const vector<double> &targetVals) {
	


	// Calulate overall net errors (RMS("Root Mean Square Error") of output neuron errors)

	Layer &outputLayer = m_layers.back(); //  the last layer 
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) { // not includint the bias	
		double delta = targetVals[n] - outputLayer[n].getOutputVal(); // delta = (target[i] - actual[i])
		m_error += delta *delta; // Summing delta^2
	}
	m_error /= outputLayer.size() - 1; // m_error / n
	m_error = sqrt(m_error);			// sqrt(m-error) RMS

	// Implement a rescent average measurement
	// (it`ll nothing to do with neural net)
	// it`ll help print out an error indication of how well 
	// the net has been doing for the last several traing samples


	m_recentAverageError = 
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradiants

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers 

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradient(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer
	// update connection weights


	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}

};

void Net::feedForward(const vector<double> &inputVals) {

	// TODO 
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assighn (latch) the input values into neurons 
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);

	}

	// foorward propagation
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}


}

Net::Net(vector<unsigned>& topology)
{
	// we need to know a nums of layers
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		
		unsigned numOutputs = layerNum == topology.size() - 1	? 0 : topology[layerNum + 1];
		// we`ve made a new layer, now fill it ith neurons 
		// and a bias neuron to the layer

		// we write neuronNum <= topology[], bacause we need an additional neuron - bias
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			// Creating a new Neuron object and adding it into the layer

			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a new Neuron!" << endl;
		}
		// Force the bias node`s output value to 1.0. It`s the kast neuron created above
		m_layers.back().back().setOutputVal(1.0);

	}
}

void showVectorVals(string label, vector<double> &v, ofstream &fout) {

	
	cout << label << " ";
	fout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
		fout << v[i] << " ";
	}
	cout << endl;
	fout << endl;
	//fout.close();
}

int main() {
	TrainingData trainingData("trainingData.txt");
	ofstream fout("out.txt");
	vector<unsigned> topology;
	trainingData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	
	while (!trainingData.isEof()) {
		++trainingPass;
		cout << endl << "Pass" << trainingPass;
		fout << endl << "Pass" << trainingPass;
		// get new input data and feed it forward
		if (trainingData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals, fout);
		myNet.feedForward(inputVals);

		// collect the et`s actual results 
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals, fout);

		// Train the net what outputs should have been 
		trainingData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals, fout);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how will training is working averaged over recent 
		cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << endl;
		fout << "Net recent average error: "
			<< myNet.getRecentAverageError() << endl;
	}
	cout << "Done." << endl;
	fout.close();
	system("pause");
	return 0;
}
