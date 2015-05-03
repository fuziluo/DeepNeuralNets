package com.changjinxiong.deepneuralnets.nn;

import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.*;

/**
 * A fully connected layer in MLP
 * @author jxchang
 *
 */
public class FullyConnectedLayer implements Layer{
	private final boolean bias;
	private final int numOfPerceptron;
	private float[] activations; //the activations of the perceptrons in batch
	private final float[] weights; //the weights used to compute activations of this layer
	private float[] error; //error for backpropagation
	private final float[] gradients; 
	private final Layer previousLayer;
	private Layer nextLayer;
	private int batchSize = 0; //batch size could change in different calculation
	
	public FullyConnectedLayer(int numOfPerceptron, Layer previousLayer, Layer nextLayer, boolean bias) {
		this.bias = bias;
		this.numOfPerceptron = numOfPerceptron;
		this.previousLayer = previousLayer;
		this.nextLayer = nextLayer;
		
		if (previousLayer != null) { 
			int weightLength = previousLayer.getNumOfNodes() + (previousLayer.hasBias() ? 1 : 0);
			weights = new float[numOfPerceptron * weightLength];
			//randomly initialize weights
			initializeWeights(weights);
			gradients = new float[weights.length];
		} else {
			weights = null;
			gradients = null;
		}

	}

	@Override
	public float[] getWeight() {
		return weights;
	}

	//only for test purpose
//	public void setWeight(float[] weights) {
//		if (previousLayer == null) return;
//		if (this.weights.length != weights.length) {
//			throw new IllegalArgumentException("weights size does not match!");
//		}
//		this.weights = weights;
//	}

	@Override
	public void backpropagation() {
		if (previousLayer == null) { //input layer
			assert false; //not supposed to be here
			return;
		}
		
		if (nextLayer == null) { //output layer
//			System.out.println(Arrays.toString(error));

			//assume error has been updated by setError(float[] error)
			int weightSize = weights.length / numOfPerceptron;
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < numOfPerceptron; j++) {
					for (int k = 0; k < weightSize; k++) {
						gradients[j * weightSize + k] += error[i * numOfPerceptron + j] * previousLayer.getActivations()[i * weightSize + k];
					}
				}
			}			

		} else {
			error = new float[numOfPerceptron * batchSize];
			fullyConnectedBackPropagationCalculate(nextLayer.getError(), nextLayer.getWeight(), previousLayer.getActivations(), activations, error, weights, gradients);
		}
		
	}
	
	private void fullyConnectedBackPropagationCalculate(float[] nextError, float[] nextWeights,
			float[] previousActivations, float[] activations, 
			float[] error, float[] weights, float[] weightsDerivative) {
		int nextWeightsStep = bias? (numOfPerceptron + 1) : numOfPerceptron;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				float activation = activations[i * (numOfPerceptron + (bias ? 1 : 0)) + j];
				float derivative = activation * (1 - activation);
				for (int k = 0; k < nextLayer.getNumOfNodes(); k++) {
					error[i * numOfPerceptron + j] += nextWeights[k * nextWeightsStep + j] * nextError[i * nextLayer.getNumOfNodes() + k];
				}
				error[i * numOfPerceptron + j] *= derivative;
			}
		}
//		System.out.println(Arrays.toString(error));
		int weightSize = weights.length / numOfPerceptron;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				for (int k = 0; k < weightSize; k++) {
					weightsDerivative[j * weightSize + k] += error[i * numOfPerceptron + j] * previousActivations[i * weightSize + k];
				}
			}
		}

		
	}

	@Override
	public float[] getActivations() {
		return activations;
	}

	@Override
	public void setActivations(float[] activations) {
		if (previousLayer != null) {
			assert false; //not supposed to be here
			return;
		}
		if (activations.length % (numOfPerceptron + (bias ? 1 : 0)) != 0) {
			throw new IllegalArgumentException("activations size error!");
		}
		batchSize = activations.length / (numOfPerceptron + (bias ? 1 : 0));
		this.activations = activations;
		
	}

	@Override
	public void forwardPass() {
		if (previousLayer == null) { //input layer
			assert false; //not supposed to be here
			return;
		}
		batchSize = previousLayer.getBatchSize(); //update batch size
		if (bias) {
			activations = new float[batchSize * (numOfPerceptron + 1)];
		} else {
			activations = new float[batchSize * numOfPerceptron];
		}
		fullyConnectedForwardPassCalculate(previousLayer.getActivations(), weights, activations);		
	}

	private void initializeWeights(float[] weights) {
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextFloat();
		}
	}

	private void fullyConnectedForwardPassCalculate(float[] previousActivations,
			float[] weights, float[] activations) {
		int activationStep = bias? (numOfPerceptron + 1) : numOfPerceptron;
		int weightsStep = weights.length/numOfPerceptron;
		for (int i = 0; i < previousLayer.getBatchSize(); i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				for (int k = 0; k < weightsStep; k++) {
					activations[i*activationStep + j] += weights[j*weightsStep + k]*previousActivations[i*weightsStep + k];
				}
				activations[i*activationStep + j] = (float) (1.0/(1 + exp(-activations[i*activationStep + j])));
			}
			if (bias) {
				activations[i*activationStep + numOfPerceptron] = 1; //bias node
				
			}
		}
	}

	@Override
	public Layer getPreviousLayer() {
		return previousLayer;
	}

//	@Override
//	public void setPreviousLayer(Layer previousLayer) {
//		this.previousLayer = previousLayer;
//	}

	@Override
	public Layer getNextLayer() {
		return nextLayer;
	}

	@Override
	public void setNextLayer(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}

	@Override
	public boolean hasBias() {
		return bias;
	}

	@Override
	public float[] getError() {
		if (error == null) {
			assert false; //not supposed to be here
		}
		return error;
	}
	
	@Override
	public void setError(float[] error) {
		this.error = error;
	}

	@Override
	public int getBatchSize() {
		return batchSize;
	}

	@Override
	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	@Override
	public int getNumOfNodes() {
		return numOfPerceptron;
	}
	
	@Override
	public float[] getGradients() {
		return gradients;
	}

	public void updateWeights(float learningRate) {
		if (previousLayer == null) { //input layer
			assert false; //not supposed to be here
			return;
		}
		for (int i = 0; i < weights.length; i++) {
//			System.out.println("!"+weights[i]+" "+gradients[i]);
			weights[i] -= learningRate * gradients[i];
			gradients[i] = 0;
		}
	}

}
