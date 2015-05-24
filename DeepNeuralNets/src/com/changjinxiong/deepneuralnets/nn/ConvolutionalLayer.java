package com.changjinxiong.deepneuralnets.nn;

import java.util.Arrays;
import java.util.Random;

import static java.lang.Math.*;

public class ConvolutionalLayer implements FeatureMapLayer {
	public enum ActivationFunction {SIGMOID, RELU, SOFTPLUS, TANH}
	private final boolean addBias;
	private final FeatureMapLayer previousLayer;
	private Layer nextLayer;
	private int batchSize = 1; //batch size could change in different calculation

	private final int numOfInputFeatureMaps;
	private final int numOfOutputFeatureMaps;
	private final int stride;
	private final int filterHeight;
	private final int filterWidth;
	
	private int[] inputFeatureMapsShape; //height, width
	private int[] outputFeatureMapsShape; //height, width
	
	private float[] activations; //the output feature map
	private float[] weights;
	private float[] errors; //error used for calculating gradients in backpropagation, get from next layer or set by MLP
	private float[] prevErrors; // error in the previous layer, calculated in this layer
	private final float[] gradients; 
	private final float[] weightsUpdate; 		
	private ActivationFunction activationFunction;

	public ConvolutionalLayer(int numOfOutputFeatureMaps, int filterHeight, int filterWidth, int stride, 
			FeatureMapLayer previousLayer, Layer nextLayer, boolean addBias) {
		this.previousLayer = previousLayer;
		this.nextLayer = nextLayer;
		this.numOfOutputFeatureMaps = numOfOutputFeatureMaps;
		if (previousLayer != null) {
			//TODO add support to subsampling layer
			this.numOfInputFeatureMaps = previousLayer.getNumOfFeatureMaps();
			//the last element of each weight is bias.
			this.weights = new float[numOfOutputFeatureMaps * (filterHeight * filterWidth + (addBias ? 1 : 0))];
			initializeWeights(weights);
			this.gradients = new float[weights.length];
			this.weightsUpdate = new float[weights.length];
			//TODO change the default setting
			activationFunction = ActivationFunction.SIGMOID;
		} else {
			if (filterHeight != 0 || filterWidth != 0 || stride != 0 ) {
				throw new IllegalArgumentException("filterHeight, filterWidth and stride must be 0 if the layer is input layer.");
			}
			if (addBias == true) {
				throw new IllegalArgumentException("addBias should be false in input layer.");
			}
			this.numOfInputFeatureMaps = 0;
			this.weights = null;
			this.gradients = null;
			this.weightsUpdate = null;
		}
		this.addBias = addBias;
		this.filterHeight = filterHeight;
		this.filterWidth = filterWidth;
		this.stride = stride;


	}
	//TODO use for old handling of bias in fully connected layer, can be improved in future
	private boolean addBiasNode() {
		if (getNextLayer() != null && getNextLayer() instanceof FullyConnectedLayer) {
			return getNextLayer().hasBias();
		} else {
			return false;
		}
	}
	
	private void initializeWeights(float[] weights) {
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextFloat() - 0.5f;
		}
	}
	private float activationFunction(float input) {
		float output;
		switch (activationFunction) {
		case RELU:
			output = max(0, input);
			break;
		case SOFTPLUS:
			output = (float) log(1 + exp(input));
			break;
		case SIGMOID:
			output = (float) (1/(1 + exp(-input)));
			break;
		case TANH:
			output = (float) ((1 - exp(-input)) / (1 + exp(-input)));
			break;
		default:
			output = (float) (1/(1 + exp(-input)));
			break;
		}
		return output;
	}
	public void setActivationFunction(ActivationFunction func) {
		if (previousLayer == null) { 
			throw new IllegalStateException("No Activation Function on input layer!");
		}	
		activationFunction = func;
	}
	
	@Override
	public int getNumOfFeatureMaps() {
		return numOfOutputFeatureMaps;
	}
	
	@Override
	public void updateFeatureMapsShapes() {
		if (previousLayer == null) { 
			throw new IllegalStateException("Input layer shouldn't call updateFeatureMapsShapes()!");
		}
		inputFeatureMapsShape = previousLayer.getOutputFeatureMapsShapes();
		//calculating output feature map size from input feature map size
		int h = (inputFeatureMapsShape[0] - filterHeight) / stride + 1;
		int w = (inputFeatureMapsShape[1] - filterWidth) / stride + 1;
		outputFeatureMapsShape = new int[] {h, w};
	}
	
	@Override
	public int[] getOutputFeatureMapsShapes() {
		if (previousLayer != null) { 
			updateFeatureMapsShapes();
		}
		return outputFeatureMapsShape;
	}

	@Override
	public float[] getWeight() {
		if (previousLayer == null) { 
			throw new IllegalStateException("No weights on input layer!");
		}	
		return weights;	
	}

	@Override
	public void setWeight(float[] weights) {
		if (previousLayer == null) {
			throw new IllegalStateException("Cannot set weight on input layer!");
		}
		if (this.weights != null && this.weights.length != weights.length) {
			throw new IllegalArgumentException("weights size does not match!");
		}
		this.weights = weights;
	}
	
	@Override
	public void updateWeights(float learningRate, float momentum) {
		if (previousLayer == null) { 
			throw new IllegalArgumentException("Not allowed to update weight on input layer!");
		}
		for (int i = 0; i < weights.length; i++) {
			weightsUpdate[i] = momentum * weightsUpdate[i] - learningRate * gradients[i];
			weights[i] += weightsUpdate[i];
			gradients[i] = 0;
		}
	}

	@Override
	public void backpropagation(boolean useOpenCL) {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not backpropagation calculation on input layer!");
		}
		if (nextLayer == null) { //output layer
			//assume error has been updated by setError(float[] error)
		} else {
			errors = nextLayer.getPrevErrors();
		}
		if (useOpenCL) {
			backPropOpenCL();
		} else {
			backPropNoAcc();
		}
	}

	private void backPropNoAcc() {
		/**************************************
		 * calculating gradients
		 **************************************/
		float[] inputFeatureMaps = previousLayer.getActivations();
		int inputFeatureMapSize = inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
		int outputFeatureMapSize = outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
//		int activationDim = outputFeatureMapSize + (addBiasNode()? 1 : 0);
		int weightsDim  = filterHeight * filterWidth + (addBias ? 1 : 0);	
		for (int i = 0; i < batchSize; i++) {
			int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapSize;
			int batchOffsetOut = i * numOfOutputFeatureMaps * outputFeatureMapSize;
			for (int j = 0; j < numOfOutputFeatureMaps; j++) {
				int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapSize;
				for (int row = 0, col = 0; col + filterWidth <= inputFeatureMapsShape[1]; row += stride) {
					if (row + filterHeight > inputFeatureMapsShape[0]) {
						col += stride;
						if (col + filterWidth > inputFeatureMapsShape[1]) {
							break;
						}
						row = 0;
					}		
					int rowIndOut = row / stride;
					int colIndOut = col / stride;
					for (int k = 0; k < numOfInputFeatureMaps; k++) {
						int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapSize;
						/****************************************/
						//TODO the loops can be re-arranged to minimize the computational cost of averaging over batchSize
						for (int m = 0; m < weightsDim - 1; m++) {
							int rowIndIn = m / filterWidth + row;
							int colIndIn = m % filterWidth + col;
							gradients[j *  weightsDim + m] += errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] * 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] / batchSize;
						}
						if (!addBias) { 
							int rowIndIn = (weightsDim - 1) / filterWidth + row;
							int colIndIn = (weightsDim - 1) % filterWidth + col;
							gradients[j *  weightsDim + weightsDim - 1] += errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] * 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] / batchSize;
						} else { //for the last weight, aka bias
							gradients[j *  weightsDim + weightsDim - 1] += errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] / batchSize;
						}	
						/****************************************/
					}
				}
			}
		}
		
		/**************************************
		 * calculating previous error
		 **************************************/		
		if (previousLayer.getPreviousLayer() == null) {
			return;
		}
		prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
		for (int i = 0; i < batchSize; i++) {
			int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapSize;
			int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapSize + (addBiasNode()? 1 : 0));
			for (int j = 0; j < numOfOutputFeatureMaps; j++) {
				int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapSize;
				for (int row = 0, col = 0; col + filterWidth <= inputFeatureMapsShape[1]; row += stride) {
					if (row + filterHeight > inputFeatureMapsShape[0]) {
						col += stride;
						if (col + filterWidth > inputFeatureMapsShape[1]) break;
						row = 0;
					}		
					int rowIndOut = row / stride;
					int colIndOut = col / stride;
					for (int k = 0; k < numOfInputFeatureMaps; k++) {
						int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapSize;
						/****************************************/
						for (int m = 0; m < filterHeight * filterWidth; m++) {
							int rowIndIn = m / filterWidth + row;
							int colIndIn = m % filterWidth + col;
							prevErrors[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] += 
									weights[j *  weightsDim + m] * errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut];
						}
						
						/****************************************/
					}
				}
			}
		}


	}
	private void backPropOpenCL() {
		// TODO Auto-generated method stub
		
	}
	@Override
	public float[] getActivations() {
		return activations;
	}

	@Override
	public void setInputs(float[] inputs) {
		if (previousLayer != null) {
			throw new IllegalStateException("Only allow to set activations on input layer!");
		}
		if (outputFeatureMapsShape == null) {
			throw new IllegalStateException("set input shape first!");
		}
		if (inputs.length % (numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]) != 0) {
			throw new IllegalArgumentException("inputs size error!");
		}
		this.batchSize = inputs.length / (numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]);
//		outputFeatureMapsShape = activations.length / (numOfOutputFeatureMaps * batchSize) - (addBiasNode() ? 1 : 0);
		this.activations = inputs;
	}

	@Override
	public void forwardPass(boolean useOpenCL) {
		// TODO subsampling to be added..
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not forward pass calculation on input layer!");
		}
		batchSize = previousLayer.getBatchSize(); //update batch size
		updateFeatureMapsShapes();
		activations = new float[batchSize * (numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1] 
								+ (addBiasNode() ? 1 : 0))];
		if (useOpenCL) {
			forwardPassOpenCL();
		} else {
			forwardPassNoAcc();
		}
	}

	private void forwardPassOpenCL() {
		// TODO Auto-generated method stub
		
	}

	private void forwardPassNoAcc() {
		float[] inputFeatureMaps = previousLayer.getActivations();
		int inputFeatureMapSize = inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
		int outputFeatureMapSize = outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
//		int activationDim = outputFeatureMapSize + (addBiasNode()? 1 : 0);
		int weightsDim  = filterHeight * filterWidth + (addBias ? 1 : 0);	
		for (int i = 0; i < batchSize; i++) {
			int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapSize;
			int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapSize + (addBiasNode()? 1 : 0));
			for (int j = 0; j < numOfOutputFeatureMaps; j++) {
				int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapSize;
				for (int row = 0, col = 0; col + filterWidth <= inputFeatureMapsShape[1]; row += stride) {
					if (row + filterHeight > inputFeatureMapsShape[0]) {
						col += stride;
						if (col + filterWidth > inputFeatureMapsShape[1]) break;
						row = 0;
					}		
					int rowIndOut = row / stride;
					int colIndOut = col / stride;
					for (int k = 0; k < numOfInputFeatureMaps; k++) {
						int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapSize;
						for (int m = 0; m < weightsDim - 1; m++) {
							int rowIndIn = m / filterWidth + row;
							int colIndIn = m % filterWidth + col;
													
							//the handling of convolution here is simply weighted sum, considering the order of weights is reversed.
							activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] += 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] * weights[j *  weightsDim + m];
						}
						if (!addBias) { 
							int rowIndIn = (weightsDim - 1) / filterWidth + row;
							int colIndIn = (weightsDim - 1) % filterWidth + col;
							activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] += 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] * weights[j *  weightsDim + weightsDim - 1];
	
						} else { 
							activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] += weights[j *  weightsDim + weightsDim - 1];
						}			
					}
					activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] = 
							activationFunction(activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut]);
				}
			}
			//TODO for the next fully connected layer. Can be optimized in future
			if (addBiasNode()) {
				activations[batchOffsetOut + numOfOutputFeatureMaps * outputFeatureMapSize] = 1;
			}
		}
	}

	@Override
	public Layer getPreviousLayer() {
		return previousLayer;
	}

	@Override
	public Layer getNextLayer() {
		return nextLayer;
	}

	@Override
	public void setNextLayer(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}

	@Override
	public float[] getErrors() {
		if (previousLayer == null) { //not input layer
			throw new IllegalStateException("No error on input layer!");
		}		
		return errors;
	}

	@Override
	public float[] getGradients() {
		if (previousLayer == null) { //not input layer
			throw new IllegalStateException("No gradients on input layer!");
		}	
		return gradients;
	}

	@Override
	public void setErrors(float[] error) {
		if (nextLayer != null) { //not output layer
			throw new IllegalStateException("only allow to set error on output layer!");
		}	
		this.errors = error;
	}

	@Override
	public int getBatchSize() {
		return batchSize;
	}

	@Override
	public void setInputShape(int[] inputShape) {
		if (previousLayer != null) { //not input layer
			throw new IllegalStateException("only allow to set batch size on input layer!");
		}
		if (inputShape.length != 2) {
			throw new IllegalArgumentException("Convolutional layer has 2 shape parameters for input!");
		}
		outputFeatureMapsShape = new int[2];
		outputFeatureMapsShape[0] = inputShape[0];
		outputFeatureMapsShape[1] = inputShape[1];
		
	}

	@Override
	public boolean hasBias() {
		return addBias;
	}

	@Override
	public int getNumOfNodes() {
		if (outputFeatureMapsShape == null) {
			updateFeatureMapsShapes();
		}
		int height = outputFeatureMapsShape[0];
		int width = outputFeatureMapsShape[1];
		return numOfOutputFeatureMaps * (height * width);

	}
	@Override
	public float[] getPrevErrors() {
		if (previousLayer == null|| previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		return prevErrors;
	}

}
