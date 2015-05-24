package com.changjinxiong.deepneuralnets.nn;

import com.changjinxiong.deepneuralnets.test.DataProvider;

public interface NeuralNetwork {

	public Layer getInputLayer();

	public Layer getOutputLayer();

	public void fordwardPass(float[] inputSamples, boolean useOpenCL);

	public void backPropagation(float[] labels, int costType, boolean useOpenCL);

	public void updateWeights(float learningRate, float momentum);

	public float test(DataProvider dp, boolean useOpenCL);

	public void train(DataProvider dp, int costType, float learningRate, float momentum, int decayCycle, float decayRate,
			int maxEpoch, boolean useOpenCL);
	public float getCost(float[] labels, int costType);

	public void setError(float[] labels, int costType);

}