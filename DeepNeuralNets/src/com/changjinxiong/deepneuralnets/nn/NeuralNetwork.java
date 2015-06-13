package com.changjinxiong.deepneuralnets.nn;

import com.changjinxiong.deepneuralnets.test.DataProvider;

public interface NeuralNetwork {

	public Layer getInputLayer();

	public Layer getOutputLayer();

	public void fordwardPass(float[] inputSamples);

	public void backPropagation(float[] labels, int costType);

	public void updateWeights(float learningRate, float momentum, float weightDecay);

	public float test(DataProvider dp);

	public void train(DataProvider dp, int costType, float learningRate, float momentum, float weightDecay, int lrChangCycle, float lrChangRate,
			int maxEpoch);
	public float getCost(float[] labels, int costType);

	public void setError(float[] labels, int costType);
	
	public void saveWeights(String path);
	
	public void loadWeights(String path);
	
}