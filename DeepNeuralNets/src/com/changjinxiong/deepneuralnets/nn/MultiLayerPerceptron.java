package com.changjinxiong.deepneuralnets.nn;
import static java.lang.Math.*;

import java.util.ArrayList;
import java.util.Arrays;

import com.changjinxiong.deepneuralnets.opencl.OpenCL;
import com.changjinxiong.deepneuralnets.test.DataProvider;
import com.changjinxiong.deepneuralnets.test.IrisDataProvider;


/**
 * Multi-layer Perceptron
 * @author jxchang
 *
 */
public class MultiLayerPerceptron implements NeuralNetwork {
	
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
		inputLayer = new FullyConnectedLayer(perceptronsOfLayers[0], null, null, false);
		outputLayer = inputLayer;		
		for (int i = 1; i < numOfLayers; i++ ) {
			Layer newLayer = new FullyConnectedLayer(perceptronsOfLayers[i], outputLayer, null, bias);
			outputLayer.setNextLayer(newLayer);
			outputLayer = newLayer;
		}		
	}
	
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#getInputLayer()
	 */
	@Override
	public Layer getInputLayer() {
		return inputLayer;
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#getOutputLayer()
	 */
	@Override
	public Layer getOutputLayer() {
		return outputLayer;
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#fordwardPass(float[], int, boolean)
	 */
	@Override
	public void fordwardPass(float[] inputSamples, int batchSize, boolean useOpenCL) {
		inputLayer.setInputShape(new int[] {batchSize});
		inputLayer.setInputs(inputSamples); //provide input data
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			currentLayer.forwardPass(useOpenCL);
		}
				
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#backPropagation(float[], boolean)
	 */
	@Override
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
//				error[i * labelSize + j] *= activations[i * labelSize + j] * (1 - activations[i * labelSize + j]);
			}
		}
//		System.out.println(Arrays.toString(activations));
//		System.out.println(Arrays.toString(labels));
		//set the error
		outputLayer.setErrors(error); 
		Layer currentLayer = outputLayer;
		while (currentLayer.getPreviousLayer() != null) {
			currentLayer.backpropagation(useOpenCL);
			currentLayer = currentLayer.getPreviousLayer();
		}		
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#updateWeights(float, float)
	 */
	@Override
	public void updateWeights(float learningRate, float momentum) {
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			currentLayer.updateWeights(learningRate, momentum);
		}
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#test(com.changjinxiong.deepneuralnets.test.DataProvider, boolean)
	 */
	@Override
	public float test(DataProvider dp, boolean useOpenCL) {
		int testNum = 0;
		ArrayList<Float> testResult = new ArrayList<Float>();
		ArrayList<Float> labels = new ArrayList<Float>();
		for ( ; testNum < dp.getDatasetSize(); testNum += dp.getBatchSize()) {
//		for ( ; testNum < 1; testNum += dp.getBatchSize()) {
			fordwardPass(dp.getNextbatchInput(bias), dp.getBatchSize(), useOpenCL);
			for (float a : getOutputLayer().getActivations()) {
				testResult.add(a);
			}
			for (float l : dp.getNextBatchLabel()) {
				labels.add(l);
			}
		}
//		System.out.println(testNum);
		Float[] a = new Float[testNum * dp.getLabelDimension()];
		a = testResult.toArray(a);
		Float[] t = new Float[testNum * dp.getLabelDimension()];
		t = labels.toArray(t);
		float count = 0;
		for (int i = 0; i < testNum * dp.getLabelDimension(); i += dp.getLabelDimension()) {
			int maxInd = 0;
			for (int j = 0; j < dp.getLabelDimension() - 1; j++) {
				if (a[i + maxInd] > a[i + j + 1]) {
					a[i + j + 1] = 0f;
				} else {
					a[i + maxInd] = 0f;
					maxInd = j + 1;
				}
			}
			a[i + maxInd] = 1f;
			if (t[i + maxInd] != 1){
				count++;
			}
		}
		
//		System.out.println(Arrays.toString(a));
//		System.out.println(Arrays.toString(t));
		float errorRate = count/testNum;
		System.out.printf("%.0f out of %d wrong. Error rate is %.2f\n", count, testNum, errorRate);
		return errorRate;
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#train(com.changjinxiong.deepneuralnets.test.DataProvider, float, float, int, boolean)
	 */
	@Override
	public void train(DataProvider dp, float learningRate, float momentum, 
			 int decayCycle, float decayRate, int maxEpoch, boolean useOpenCL) {
		if (learningRate < 0) {
			throw new IllegalArgumentException("learningRate cannot be negtive");
		}
		if (momentum < 0) {
			throw new IllegalArgumentException("momentum cannot be negtive");
		}
		if (decayCycle < 0) {
			throw new IllegalArgumentException("decayCycle cannot be negtive");
		}
		if (decayCycle > 0 && (decayRate <= 0 || decayRate >= 1)) {
			throw new IllegalArgumentException("decayRate should be within (0, 1)");
		}
		dp.reset();
		float baseLr = learningRate;
		float averageCost = 0;
		for (int i = 0, j = 0, k = 0; i < dp.getDatasetSize() * maxEpoch; i += dp.getBatchSize(), j++, k++) {
//		for (int i = 0; i <  1; i += dp.getBatchSize()) {
			fordwardPass(dp.getNextbatchInput(bias), dp.getBatchSize(), useOpenCL);
			//monitor the cost
			float [] batchLabels = dp.getNextBatchLabel();
			float cost = getCost(batchLabels);
//			System.out.println(cost);
			backPropagation(batchLabels, useOpenCL);
			averageCost += cost;
			if (k >= dp.getDatasetSize() / dp.getBatchSize()) {
				averageCost /= k;
				System.out.printf("Average cost over last %d batches is %.5f\n", k, averageCost);
				k = 0;
				averageCost = 0;
			}
			if (decayCycle > 0 && j >= decayCycle) {
				//TODO add argument to control decay
//				baseLr *= 0.8;
				baseLr *= decayRate;
				System.out.printf("learning rate reduced to %f after %d batches\n", baseLr, i);
				j = 0;
			}
			updateWeights(baseLr, momentum);	
			//////////////////////////
//			return;
		}
		if (useOpenCL) {
			OpenCL.releaseAll();
		}
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#getCost(float[])
	 */
	@Override
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
				/************************************************
				 * Compute cost with activation computed in fordwardPass.
				 * cost function used: logistic regression cost function without regularization.
				 * J = -sum(y*log(a)+(1-y)*log(1-a))/m, where y is class label, a is sigmoid activation, 
				 * m is the number of samples. 
				 * 
				 ************************************************/
				singleCost += (labels[j * labelSize + i] == 1) ? log(activations[j * labelSize + i]) : log(1 - activations[j * labelSize + i]);
				/************************************************
				 * MSE cost
				 ************************************************/
//				singleCost += -0.5*pow(labels[j * labelSize + i] - activations[j * labelSize + i], 2);
			}
			result -= singleCost/batchSize;
		}

		return result;
	}

}
