package com.changjinxiong.deepneuralnets.nn;

public interface WeightLayer extends Layer{
	/**
	 * Method used to update the weight of this layer in each iteration
	 * @param learningRate
	 * @param momentum
	 * @param weightDecay
	 */
	public void updateWeights(float learningRate, float momentum, float weightDecay);
	/**
	 * Method used to load provided weight to the layer.
	 * @param weights
	 */
	public void setWeight(float[] weights);
	/**
	 * 
	 * @return the weights of the current layer as a array
	 */
	public float[] getWeight();
	/**
	 * Initialize the weight using uniform distribution. Initialize the bias using the given value
	 * @param lowerLimit lower limit of the uniform distribution
	 * @param upperLimit upper limit of the uniform distribution
	 * @param bias the value used to set all the biases
	 */
	public void initWeightsUniform(float lowerLimit, float upperLimit, float bias);
	/**
	 * Initialize the weight using Gaussian distribution. Initialize the bias using the given value
	 * @param std the standard deviation
	 * @param mean the mean
	 * @param bias the value used to set all the biases
	 */
	public void initWeightsGaussian(float std, float mean, float bias);
	/**
	 * Initialize the weight using Gaussian distribution while automatically choose the mean and standard deviation.
	 * Initialize the bias using the given value
	 * @param bias the value used to set all the biases
	 */
	public void initWeights(float bias);
}
