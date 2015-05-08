package com.changjinxiong.deepneuralnets.nn;
import static java.lang.Math.*;

import java.util.Arrays;

import com.changjinxiong.deepneuralnets.opencl.OpenCL;


/**
 * Multi-layer Perceptron
 * @author jxchang
 *
 */
public class MultiLayerPerceptron {
	
	private Layer inputLayer = null;
	private Layer outputLayer = null;
	private final boolean bias;
	
	
	public MultiLayerPerceptron(int[] perceptronsOfLayers, boolean bias) {
		if (perceptronsOfLayers.length < 2) {
			throw new IllegalArgumentException("at least 2 layers.");
		}
		for (int num : perceptronsOfLayers) {
			if (num < 1) {
				throw new IllegalArgumentException("each layer must have positive number of perceptrons.");
			}
		}
		this.bias = bias;
		int numOfLayers = perceptronsOfLayers.length;
		inputLayer = new FullyConnectedLayer(perceptronsOfLayers[0], null, null, bias);
		outputLayer = inputLayer;		
		for (int i = 1; i < numOfLayers - 1; i++ ) {
			Layer newLayer = new FullyConnectedLayer(perceptronsOfLayers[i], outputLayer, null, bias);
			outputLayer.setNextLayer(newLayer);
			outputLayer = newLayer;
		}
		Layer newLayer = new FullyConnectedLayer(perceptronsOfLayers[numOfLayers - 1], outputLayer, null, false);
		outputLayer.setNextLayer(newLayer);
		outputLayer = newLayer;

		
		
	}
	
	public Layer getInputLayer() {
		return inputLayer;
	}
	public Layer getOutputLayer() {
		return outputLayer;
	}
	public void fordwardPass(float[] inputSamples, boolean useOpenCL) {
		inputLayer.setActivations(inputSamples); //provide input data
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			currentLayer.forwardPass(useOpenCL);
		}
				
	}
	public void backPropagation(float[] labels, boolean useOpenCL) {
		//calculate the error aka the derivative of mean squared error cost function
		int labelSize = outputLayer.getNumOfNodes();
		int batchSize = outputLayer.getBatchSize();
		float[] activations = outputLayer.getActivations();
		float[] error = new float[labelSize * batchSize];
		//calculate the error
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < labelSize; j++) {
				error[i * labelSize + j] = (-labels[i * labelSize + j] + activations[i * labelSize + j]);
			}
		}
//		System.out.println(Arrays.toString(activations));
		//set the error
		outputLayer.setError(error); 
		Layer currentLayer = outputLayer;
		while (currentLayer.getPreviousLayer() != null) {
			currentLayer.backpropagation(useOpenCL);
			currentLayer = currentLayer.getPreviousLayer();
		}		
	}
	public void updateWeights(float learningRate) {
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			currentLayer.updateWeights(learningRate);
		}
	}
	public void train(TrainingDataProvider trainingDataProvider, float learningRate, int maxEpoch, boolean useOpenCL) {
		trainingDataProvider.reset();
		for (int i = 0; i < trainingDataProvider.getDatasetSize() * maxEpoch; i += trainingDataProvider.getBatchSize()) {
			fordwardPass(trainingDataProvider.getNextbatchInput(bias), useOpenCL);
			//monitor the cost
			float [] batchLabels = trainingDataProvider.getNextBatchLabel();
			float cost = getCost(batchLabels);
			System.out.println(cost);
			backPropagation(batchLabels, useOpenCL);
			updateWeights(learningRate);	
			//////////////////////////
//			return;
		}
		if (useOpenCL) {
			OpenCL.releaseAll();
		}
	}
	/**
	 * Compute cost with activation computed in fordwardPass.
	 * cost function used: logistic regression cost function without regularization.
	 * J = -sum(y*log(a)+(1-y)*log(1-a))/m, where y is class label, a is sigmoid activation, 
	 * m is the number of samples. 
	 * @return the sum of cost of all output nodes
	 */
	public float getCost(float[] labels) {
		float[] activations = outputLayer.getActivations();
		if (activations.length != labels.length) {
			throw new IllegalArgumentException("incorrect label length!");
		}
		float result = 0;
		int batchSize = outputLayer.getBatchSize();
		int labelSize = outputLayer.getNumOfNodes();
		for (int i = 0; i < labelSize; i++) {
			float singleCost = 0;
			for (int j = 0; j < batchSize; j++) {
//				singleCost += labels[j * labelSize + i] * log(activations[j * labelSize + i]) + 
//						(1 - labels[j * labelSize + i]) * log(1 - activations[j * labelSize + i]);
				//the above formula could produce NaN when activation == 1
				singleCost += (labels[j * labelSize + i] == 1) ? log(activations[j * labelSize + i]) : log(1 - activations[j * labelSize + i]);
			}
			result -= singleCost/batchSize;
		}

		return result;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
