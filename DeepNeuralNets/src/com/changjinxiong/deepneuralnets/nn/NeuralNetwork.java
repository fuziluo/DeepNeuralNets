package com.changjinxiong.deepneuralnets.nn;

import com.changjinxiong.deepneuralnets.test.DataProvider;

public interface NeuralNetwork {
	/**
	 * Get the input layer of this network
	 * @return
	 */
	public Layer getInputLayer();
	/**
	 * Get the output layer of this network
	 * @return
	 */
	public Layer getOutputLayer();
	/**
	 * Perform the forward pass calculation of this network
	 * @param inputSamples the input data
	 */
	public void forwardPass(float[] inputSamples);
	/**
	 * Perform the back propagation of this network
	 * @param labels the label corresponding to the data fed for forward pass calculation
	 * @param costType 0: cross entropy; 1: mean square error
	 */
	public void backPropagation(float[] labels, int costType);
	/**
	 * Updated the weights of all the layers in this network
	 * @param learningRate
	 * @param momentum
	 * @param weightDecay
	 */
	public void updateWeights(float learningRate, float momentum, float weightDecay);
	/**
	 * Test the whole dataset provided
	 * @param dp
	 * @return the error rate
	 */
	public float test(DataProvider dp);
	/**
	 * Train the network using the provided dataset.
	 * @param dp the dataset use for training
	 * @param costType 0: cross entropy; 1: mean square error
	 * @param learningRate the learning rate
	 * @param momentum the momentum
	 * @param weightDecay the weight decay, aka L2 regularization
	 * @param lrChangCycle defines after how many cycle of training epochs the learning rate changes
	 * @param lrChangRate defines the multiplication rate used to change the learning rate
	 * @param maxEpoch the maximum training epochs (how many iterations of the whole training set)
	 */
	public void train(DataProvider dp, int costType, float learningRate, float momentum, float weightDecay, int lrChangCycle, float lrChangRate,
			int maxEpoch);
	/**
	 * Get the cost/loss value calculated from the most recent forward pass
	 * @return the cost/loss value
	 */
	public float getCost();
	/**
	 * The method used to calculate the cost and errors in the end of every forward
	 * pass calculation. This method is only used internally or for test purpose.
	 * @param labels the label corresponding to the data fed for forward pass calculation
	 * @param costType 0: cross entropy; 1: mean square error
	 */
	public void calCostErr(float[] labels, int costType);
	/**
	 * Save the weights of all the layers to the provided path
	 * @param path
	 */
	public void saveWeights(String path);
	/**
	 * Load weights from the specified path
	 * @param path
	 */
	public void loadWeights(String path);
	/**
	 * Release all the OpenCL memory objects used in this network.
	 */
	public void releaseCLMem();

	
}