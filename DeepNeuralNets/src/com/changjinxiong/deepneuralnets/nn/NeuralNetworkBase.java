package com.changjinxiong.deepneuralnets.nn;

import static java.lang.Math.log;
import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.changjinxiong.deepneuralnets.opencl.OpenCL;
import com.changjinxiong.deepneuralnets.test.DataProvider;

public class NeuralNetworkBase implements NeuralNetwork {
	protected Layer inputLayer = null;
	protected Layer outputLayer = null;
	protected boolean addBias;
	protected final static Logger LOGGER = Logger.getLogger(FullyConnectedLayer.class.getName()); 

	@Override
	protected void finalize() {
		OpenCL.releaseAll();
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
	public void fordwardPass(float[] inputSamples) {
		inputLayer.setInputs(inputSamples); //provide input data
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			currentLayer.forwardPass();
		}
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#backPropagation(float[], boolean)
	 */
	@Override
	public void backPropagation(float[] labels, int costType) {
		setError(labels, costType);
		Layer currentLayer = outputLayer;
		while (currentLayer.getPreviousLayer() != null) {
			currentLayer.backpropagation();
			currentLayer = currentLayer.getPreviousLayer();
		}		
	}
	
	@Override
	public void setError(float[] labels, int costType) {
		//calculate the error i.e. the derivative of cost function
		int labelSize = outputLayer.getNumOfNodes();
		int batchSize = outputLayer.getBatchSize();
		float[] activations = outputLayer.getActivations();
		float[] error = new float[labelSize * batchSize];
		//calculate the error
		if (costType == 0) {
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < labelSize; j++) {
					/*****************************
					 * derivative of cross entropy
					 *****************************/
					error[i * labelSize + j] = (-labels[i * labelSize + j] + activations[i * labelSize + j]);
				}
			}
		} else if (costType == 1) {
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < labelSize; j++) {
					/*****************************************
					 * derivative of MSE
					 *****************************************/
					error[i * labelSize + j] = (-labels[i * labelSize + j] + activations[i * labelSize + j]);
					error[i * labelSize + j] *= activations[i * labelSize + j] * (1 - activations[i * labelSize + j]);
				}
			}			
		} else {
			throw new IllegalArgumentException("Incorrect cost type!");
		}
		//set the error
		outputLayer.setErrors(error); 
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#updateWeights(float, float)
	 */
	@Override
	public void updateWeights(float learningRate, float momentum, float weightDecay) {
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			currentLayer.updateWeights(learningRate, momentum, weightDecay);
		}
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#test(com.changjinxiong.deepneuralnets.test.DataProvider, boolean)
	 */
	@Override
	public float test(DataProvider dp) {
		int testNum = 0;
		ArrayList<Float> testResult = new ArrayList<Float>();
		ArrayList<Float> labels = new ArrayList<Float>();
		for ( ; testNum < dp.getDatasetSize(); testNum += dp.getBatchSize()) {
			fordwardPass(dp.getNextbatchInput(addBias));
			for (float a : getOutputLayer().getActivations()) {
				testResult.add(a);
			}
			for (float l : dp.getNextBatchLabel()) {
				labels.add(l);
			}
		}

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
		
		float errorRate = count/testNum;
		LOGGER.log(Level.INFO, "{0} out of {1} wrong. Error rate is {2}\n", new Object[] {count, testNum, errorRate} );
//		System.out.printf("%.0f out of %d wrong. Error rate is %.2f\n", count, testNum, errorRate);
		return errorRate;
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#train(com.changjinxiong.deepneuralnets.test.DataProvider, float, float, int, boolean)
	 */
	@Override
	public void train(DataProvider dp, int costType, float learningRate, float momentum, float weightDecay, 
			 int lrChangCycle, float lrChangRate, int maxEpoch) {
		if (costType < 0 || costType > 1) {
			throw new IllegalArgumentException("Wrong cost type");
		}
		if (learningRate < 0) {
			throw new IllegalArgumentException("learningRate cannot be negtive");
		}
		if (momentum < 0) {
			throw new IllegalArgumentException("momentum cannot be negtive");
		}
		if (lrChangCycle < 0) {
			throw new IllegalArgumentException("lrChangCycle cannot be negtive");
		}
		if (lrChangCycle > 0 && (lrChangRate <= 0 || lrChangRate >= 1)) {
			throw new IllegalArgumentException("lrChangRate should be within (0, 1)");
		}
		dp.reset();
		float baseLr = learningRate;
		float averageCost = 0;
		int lrDecayTimesLimit = 1;
		int lrDecayTimes = 0;
		for (int i = 0, j = 0, k = 0; i < dp.getDatasetSize() * maxEpoch; i += dp.getBatchSize(), j++, k++) {
			fordwardPass(dp.getNextbatchInput(addBias));
			//monitor the cost
			float [] batchLabels = dp.getNextBatchLabel();
			float cost = getCost(batchLabels, costType);
			backPropagation(batchLabels, costType);
			averageCost += cost;
			if (k >= dp.getDatasetSize() / dp.getBatchSize()) {
				averageCost /= k;
				LOGGER.log(Level.INFO, "Average cost over last {0} batches is {1}", new Object[] {k, averageCost});
//				System.out.printf("Average cost over last %d batches is %.5f\n", k, averageCost);
				k = 0;
				averageCost = 0;
			}
			if (lrChangCycle > 0 && j >= lrChangCycle && lrDecayTimes < lrDecayTimesLimit) {
				baseLr *= lrChangRate;
				LOGGER.log(Level.INFO, "learning rate reduced to {0} after {1} batches", new Object[] {baseLr, i});
//				System.out.printf("learning rate reduced to %f after %d batches\n", baseLr, i);
				j = 0;
				lrDecayTimes++;
			}
			updateWeights(baseLr, momentum, weightDecay);	
		}
//		if (useOpenCL) {
//			cleanOpenCLKernels();
//		}
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#getCost(float[])
	 */
	@Override
	public float getCost(float[] labels, int costType) {
		float[] activations = outputLayer.getActivations();
		if (activations.length != labels.length) {
			throw new IllegalArgumentException("incorrect label length!");
		}
		float result = 0;
		int batchSize = outputLayer.getBatchSize();
		int labelSize = outputLayer.getNumOfNodes();
		if (costType == 0) {
			for (int i = 0; i < labelSize; i++) {
				float singleCost = 0;
				for (int j = 0; j < batchSize; j++) {
	//				singleCost += labels[j * labelSize + i] * log(activations[j * labelSize + i]) + 
	//						(1 - labels[j * labelSize + i]) * log(1 - activations[j * labelSize + i]);
					//the above formula could produce NaN when activation == 1
					/************************************************
					 * Compute cost with activation computed in fordwardPass.
					 * cost function used: logistic cost function (cross entropy) without regularization
					 * J = -sum(y*log(a)+(1-y)*log(1-a))/m, where y is class label, a is sigmoid activation, 
					 * m is the number of samples. 
					 ************************************************/
					singleCost += (labels[j * labelSize + i] == 1) ? log(activations[j * labelSize + i]) : log(1 - activations[j * labelSize + i]);
				}
				result -= singleCost/batchSize;
			}
		} else if (costType == 1) {
			for (int i = 0; i < labelSize; i++) {
				float singleCost = 0;
				for (int j = 0; j < batchSize; j++) {
					/************************************************
					 * MSE cost without regularization
					 ************************************************/
					singleCost += -0.5*pow(labels[j * labelSize + i] - activations[j * labelSize + i], 2);
				}
				result -= singleCost/batchSize;
			}			
		} else {
			throw new IllegalArgumentException("Incorrect cost type!");
		}

		return result;
	}


	@Override
	public void saveWeights(String path) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void loadWeights(String path) {
		// TODO Auto-generated method stub
		
	}

}
