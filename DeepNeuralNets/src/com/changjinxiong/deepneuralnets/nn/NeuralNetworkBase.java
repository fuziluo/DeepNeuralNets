package com.changjinxiong.deepneuralnets.nn;

import static com.changjinxiong.deepneuralnets.nn.Util.activationDerivFunc;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.nio.file.StandardOpenOption.*;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.ByteChannel;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.opencl.OpenCL;
import com.changjinxiong.deepneuralnets.test.DataProvider;

public class NeuralNetworkBase implements NeuralNetwork {
	protected Layer inputLayer = null;
	protected Layer outputLayer = null;
	protected boolean addBias;
	protected final static Logger LOGGER = Logger.getLogger(NeuralNetwork.class.getSimpleName()); 
	protected float cost;
	private float[] error;

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
	public void forwardPass(float[] inputSamples) {
		inputLayer.setInputs(inputSamples); //provide input data
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			long t = System.currentTimeMillis();
			currentLayer.forwardPass();
//			System.out.println("Activations"+Arrays.toString(currentLayer.getActivations()));
//			System.out.printf("  forward %s %dms \n", currentLayer.getClass().getSimpleName(), (System.currentTimeMillis() - t));
		}
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#backPropagation(float[], boolean)
	 */
	@Override
	public void backPropagation(float[] labels, int costType) {
		long t = System.currentTimeMillis();
		calCostErr(labels, costType);
//		System.out.printf("  calc err %dms \n", (System.currentTimeMillis() - t));
		Layer currentLayer = outputLayer;
		while (currentLayer.getPreviousLayer() != null) {
			t = System.currentTimeMillis();
//			if(currentLayer.getPreviousLayer().getPreviousLayer() == null) {
//				System.out.println("prev activations "+currentLayer.getPreviousLayer().getActivations().length+Arrays.toString(Arrays.copyOf(currentLayer.getPreviousLayer().getActivations(), 1000)));
//			}

			currentLayer.backpropagation();
//			if (currentLayer.getGradients() != null) {
//				System.out.println("gradients: "+currentLayer.getClass().getSimpleName()+currentLayer.getGradients().length+Arrays.toString(Arrays.copyOf(currentLayer.getGradients(), 100)));
//				System.out.println("weights: "+currentLayer.getClass().getSimpleName()+currentLayer.getWeight().length+Arrays.toString(Arrays.copyOf(currentLayer.getWeight(), 100)));
//			}
//			if(currentLayer.getPreviousLayer().getPreviousLayer() != null)
//				System.out.println("prevErr: " + currentLayer.getClass().getSimpleName()+currentLayer.getPrevErrors().length+Arrays.toString(Arrays.copyOf(currentLayer.getPrevErrors(), 1000)));
//			System.out.printf("  back %s %dms \n", currentLayer.getClass().getSimpleName(), (System.currentTimeMillis() - t));
			currentLayer = currentLayer.getPreviousLayer();
		}		
		currentLayer.releaseCLMem();
//		System.out.println();
	}
	
//	@Override
//	public void setError(float[] labels, int costType) {
////		float[] activations = outputLayer.getActivations();
////		if (activations.length != labels.length) {
////			throw new IllegalArgumentException("incorrect label length!");
////		}
////		//calculate the error i.e. the derivative of cost function
////		ActivationType actType = outputLayer.getActivationType();
////		float[] error = null;
////		switch (actType) {
////		case RELU:
////			error = getReLUErr(labels, activations, costType);
////			break;
////		case SIGMOID:
////			error = getSigmoidErr(labels, activations, costType);
////			break;
////		case TANH:
////			error = getTanhErr(labels, activations, costType);
////			break;
////		default:
////			break;
////		
////		}
//		
//
//		//set the error
//		outputLayer.setErrors(error); 
//
//	}
//	private float[] getTanhErr(float[] labels, float[] activations, int costType) {
//		int labelSize = outputLayer.getNumOfNodes();
//		int batchSize = outputLayer.getBatchSize();
//		float[] error = new float[labelSize * batchSize];
//		//calculate the error
//		if (costType == 0) {
//			for (int i = 0; i < batchSize; i++) {
//				for (int j = 0; j < labelSize; j++) {
//					/*****************************
//					 * derivative of cross entropy
//					 *****************************/
//					error[i * labelSize + j] = (activations[i * labelSize + j] -  2 * labels[i * labelSize + j] + 1);
////					error[i * labelSize + j] = (-labels[i * labelSize + j] + (activations[i * labelSize + j] + 1) / 2);
//				}
//			}
//		} else if (costType == 1) {
//			for (int i = 0; i < batchSize; i++) {
//				for (int j = 0; j < labelSize; j++) {
//					/*****************************************
//					 * derivative of MSE
//					 *****************************************/
//					error[i * labelSize + j] = (activations[i * labelSize + j] - 2 * labels[i * labelSize + j] + 1);
//					System.out.println(activations[i * labelSize + j]);
//					error[i * labelSize + j] *= (1 - activations[i * labelSize + j] * activations[i * labelSize + j]) / 2;
//				}
//			}			
//		} else {
//			throw new IllegalArgumentException("Incorrect cost type!");
//		}
//		System.out.println("error "+Arrays.toString(error));
//
//		return error;
//	}
//	private float[] getSigmoidErr(float[] labels, float[] activations,
//			int costType) {
//		int labelSize = outputLayer.getNumOfNodes();
//		int batchSize = outputLayer.getBatchSize();
//		float[] error = new float[labelSize * batchSize];
//		//calculate the error
//		if (costType == 0) {
//			for (int i = 0; i < batchSize; i++) {
//				for (int j = 0; j < labelSize; j++) {
//					/*****************************
//					 * derivative of cross entropy
//					 *****************************/
//					error[i * labelSize + j] = (-labels[i * labelSize + j] + activations[i * labelSize + j]);
//				}
//			}
//		} else if (costType == 1) {
//			for (int i = 0; i < batchSize; i++) {
//				for (int j = 0; j < labelSize; j++) {
//					/*****************************************
//					 * derivative of MSE
//					 *****************************************/
//					error[i * labelSize + j] = (-labels[i * labelSize + j] + activations[i * labelSize + j]);
//					error[i * labelSize + j] *= activations[i * labelSize + j] * (1 - activations[i * labelSize + j]);
//				}
//			}			
//		} else {
//			throw new IllegalArgumentException("Incorrect cost type!");
//		}
//		return error;
//	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#updateWeights(float, float)
	 */
	@Override
	public void updateWeights(float learningRate, float momentum, float weightDecay) {
		Layer currentLayer = inputLayer;
		while (currentLayer.getNextLayer() != null) {
			currentLayer = currentLayer.getNextLayer();
			long t = System.currentTimeMillis();
//			System.out.println("gradients: " + currentLayer.getClass().getSimpleName()+Arrays.toString(currentLayer.getGradients()));
			currentLayer.updateWeights(learningRate, momentum, weightDecay);
//			System.out.printf("  updateWeights %dms \n", (System.currentTimeMillis() - t));
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
			forwardPass(dp.getNextbatchInput());
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
//			System.out.println(maxInd);
			a[i + maxInd] = 1f;
			if (t[i + maxInd] != 1){
				count++;
			}
		}
		
		float errorRate = count/testNum;
		LOGGER.log(Level.INFO, "{0} out of {1} wrong. Error rate is {2}\n", new Object[] {count, testNum, errorRate} );
//		System.out.printf("%.0f out of %d wrong. Error rate is %.2f\n", count, testNum, errorRate);
		releaseCLMem();
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
		LOGGER.log(Level.INFO, "Training start...");
		dp.reset();
//		boolean addBias = false;
//		if (this instanceof MultiLayerPerceptron) {
//			addBias = this.addBias;
//		}
		float lr = learningRate;
		float averageCost = 0;
		int lrDecayTimesLimit = Integer.MAX_VALUE;
		int lrDecayTimes = 0;
		for (int i = 0, j = 0, k = 0; i < dp.getDatasetSize() * maxEpoch; i += dp.getBatchSize()) {
			forwardPass(dp.getNextbatchInput());
			//monitor the cost
			long t = System.currentTimeMillis();
			float [] batchLabels = dp.getNextBatchLabel();
//			calCostErr(batchLabels, costType);
//			System.out.printf("  calc cost %dms \n", (System.currentTimeMillis() - t));
			backPropagation(batchLabels, costType);
			
			averageCost += cost;
			j++;
			k++;
			if (k >= dp.getDatasetSize() / dp.getBatchSize()) {
				averageCost /= k;
				LOGGER.log(Level.INFO, "Average cost over last {0} batches is {1}", new Object[] {k, ""+averageCost});
//				System.out.printf("Average cost over last %d batches is %.5f\n", k, averageCost);
				k = 0;
				averageCost = 0;
			}
			if (lrChangCycle > 0 && j >= lrChangCycle && lrDecayTimes < lrDecayTimesLimit) {
				lr *= lrChangRate;
				LOGGER.log(Level.INFO, "learning rate reduced to {0} after {1} batches", new Object[] {""+lr, i});
//				System.out.printf("learning rate reduced to %f after %d batches\n", baseLr, i);
				j = 0;
				lrDecayTimes++;
			}
			updateWeights(lr, momentum, weightDecay);	
//			if(i >= 1) break;
		} 
		releaseCLMem();
		LOGGER.log(Level.INFO, "Training finished");
	}
	/* (non-Javadoc)
	 * @see com.changjinxiong.deepneuralnets.nn.NeuralNetwork#getCost(float[])
	 */
	@Override
	public float getCost() {
		return cost;
	}
	
	public void calCostErr(float[] labels, int costType) {
		float[] activations = outputLayer.getActivations();
		float[] a = outputLayer.getPreviousLayer().getActivations();
//		System.out.println("penultimate activations"+Arrays.toString(a));

		if (activations.length != labels.length) {
			throw new IllegalArgumentException("incorrect label length!");
		}
		ActivationType actType = outputLayer.getActivationType();
		switch (actType) {
		case NONE:
			calcSoftmaxCostErr(labels, activations, costType);
			break;
		case RELU:
			calcSoftmaxCostErr(labels, activations, costType);
			break;
		case SIGMOID:
			calcSigmoidCostErr(labels, activations, costType);
			break;
		case TANH:
			calcTanhCostErr(labels, activations, costType);
			break;
		default:
			break;
		
		}
//		System.out.println("error"+Arrays.toString(error));
		outputLayer.setErrors(error); 
	}


	private void calcSoftmaxCostErr(float[] labels, float[] activations,
			int costType) {
		// TODO Auto-generated method stub
		int batchSize = outputLayer.getBatchSize();
		int labelSize = outputLayer.getNumOfNodes();
//		System.out.println("activations"+Arrays.toString(activations));
		// softmax layer calculation
		float[] actSoftmax = new float[activations.length];
		for (int j = 0; j < batchSize; j++) {
			float max = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < labelSize; i++) {
				max = Math.max(max, activations[j * labelSize + i]);
			}
			float sum = 0;
			for (int i = 0; i < labelSize; i++) {
				sum += Math.exp(activations[j * labelSize + i] - max);
			}
			sum = (float) (max + Math.log(sum));
//			System.out.println(sum);
			for (int i = 0; i < labelSize; i++) {
				actSoftmax[j * labelSize + i] = (float) (Math.exp(activations[j * labelSize + i] - sum));
//				System.out.println(actSoftmax[j * labelSize + i]);
			}
		}
//		System.out.println("actSoftmax"+Arrays.toString(actSoftmax));
//		System.out.println("labels"+Arrays.toString(labels));
		//calculate error for softmax layer
		cost = 0;
		error = new float[labelSize * batchSize];
		if (costType == 0) {
			for (int i = 0; i < labelSize; i++) {
				for (int j = 0; j < batchSize; j++) {
					cost -= labels[j * labelSize + i] == 1 ? log(actSoftmax[j * labelSize + i]) : 0;
					error[j * labelSize + i] = (-labels[j * labelSize + i] + actSoftmax[j * labelSize + i]);
				}
			}
		} else if (costType == 1) {
			for (int j = 0; j < batchSize; j++) {
				for (int i = 0; i < labelSize; i++) {
					cost += 0.5*pow(labels[j * labelSize + i] - actSoftmax[j * labelSize + i], 2);
					for (int k = 0; k < labelSize; k++) {
						float err = (-labels[j * labelSize + k] + actSoftmax[j * labelSize + k]);
						float derivative = actSoftmax[j * labelSize + i] * ((i == k ? 1 : 0) - actSoftmax[j * labelSize + k]);
						err *= derivative;
						error[j * labelSize + k] += err;
					}
					
				}
			}			
		} else {
			throw new IllegalArgumentException("Incorrect cost type!");
		}
		cost /= batchSize;
		for (int j = 0; j < batchSize; j++) {
			for (int i = 0; i < labelSize; i++) {
				error[j * labelSize + i] *= activationDerivFunc(outputLayer.getActivationType(), error[j * labelSize + i]);
			}
		}		
	}

	private void calcTanhCostErr(float[] labels, float[] activations, int costType) {
		cost = 0;
		int batchSize = outputLayer.getBatchSize();
		int labelSize = outputLayer.getNumOfNodes();
		error = new float[labelSize * batchSize];
		if (costType == 0) {
			for (int i = 0; i < labelSize; i++) {
				for (int j = 0; j < batchSize; j++) {
					cost -= (labels[j * labelSize + i] == 1) ? (labels[j * labelSize + i]) * log((1 + activations[j * labelSize + i])/2) :
						(1 -labels[j * labelSize + i]) * log((1 - activations[j * labelSize + i])/2);
//					cost -= (2 * labels[j * labelSize + i]) * log(1 + activations[j * labelSize + i]) + 
//							(2 - 2 * labels[j * labelSize + i]) * log(1 - activations[j * labelSize + i]);

					error[j * labelSize + i] = (activations[j * labelSize + i] - 2 * labels[j * labelSize + i] + 1) * 0.5f;

				}
			}
		} else if (costType == 1) {
			for (int i = 0; i < labelSize; i++) {
				for (int j = 0; j < batchSize; j++) {
					/************************************************
					 * MSE cost
					 ************************************************/
					cost += 0.5*pow(2 * labels[j * labelSize + i] - activations[j * labelSize + i] - 1, 2);
					
					error[j * labelSize + i] = (activations[j * labelSize + i] - 2 * labels[j * labelSize + i] + 1);
					error[j * labelSize + i] *= (1 - activations[j * labelSize + i] * activations[j * labelSize + i]) / 2;
				}
			}			
		} else {
			throw new IllegalArgumentException("Incorrect cost type!");
		}
		cost /= batchSize;

	}
	private void calcSigmoidCostErr(float[] labels, float[] activations, int costType) {
		cost = 0;
		int batchSize = outputLayer.getBatchSize();
		int labelSize = outputLayer.getNumOfNodes();
		error = new float[labelSize * batchSize];
		if (costType == 0) {
			for (int i = 0; i < labelSize; i++) {
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
					cost -= (labels[j * labelSize + i] == 1) ? log(activations[j * labelSize + i]) : log(1 - activations[j * labelSize + i]);
					error[j * labelSize + i] = (-labels[j * labelSize + i] + activations[j * labelSize + i]);
				}
			}
		} else if (costType == 1) {
			for (int i = 0; i < labelSize; i++) {
				for (int j = 0; j < batchSize; j++) {
					/************************************************
					 * MSE cost without regularization
					 ************************************************/
					cost += 0.5*pow(labels[j * labelSize + i] - activations[j * labelSize + i], 2);
					error[j * labelSize + i] = (-labels[j * labelSize + i] + activations[j * labelSize + i]);
					error[j * labelSize + i] *= activations[j * labelSize + i] * (1 - activations[j * labelSize + i]);
				}
			}			
		} else {
			throw new IllegalArgumentException("Incorrect cost type!");
		}
		cost /= batchSize;

	}
	@Override
	public void saveWeights(String path) {
//		String name = "." + this.getClass().getSimpleName() + ".weights";
		try {
			ByteChannel channel = Files.newByteChannel(Paths.get(path), WRITE, CREATE, TRUNCATE_EXISTING);
//			FileChannel channel = new RandomAccessFile(Paths.get(path,".weights").toString(), "rw").getChannel();
			Layer layer = inputLayer.getNextLayer();
			int bufferSize = 0;
			float weights[];
			do {
				if (layer instanceof PoolingLayer) {
					layer = layer.getNextLayer();
					continue; 
				}
				weights = layer.getWeight();
				bufferSize = 4 * weights.length;
				ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
				buffer.asFloatBuffer().put(weights);
				channel.write(buffer);
				layer = layer.getNextLayer();
			}
			while (layer != null);
			channel.close();
			LOGGER.log(Level.INFO, "Weights saved to " + path);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	@Override
	public void loadWeights(String path) {
//		String name = "." + this.getClass().getSimpleName() + ".weights";
		try {
			ByteChannel channel = Files.newByteChannel(Paths.get(path), READ);
//			FileChannel channel = new RandomAccessFile(Paths.get(path,".weights").toString(), "rw").getChannel();
			Layer layer = inputLayer.getNextLayer();
			int bufferSize = 0;
			float weights[];
			do {
				if (layer instanceof PoolingLayer) {
					layer = layer.getNextLayer();
					continue; 
				}
				weights = layer.getWeight();
				bufferSize = 4 * weights.length;
				ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
				buffer.clear();
				channel.read(buffer);
				buffer.flip();
				buffer.asFloatBuffer().get(weights);
				layer.setWeight(weights);
				layer = layer.getNextLayer();
			}
			while (layer != null);
			channel.close();	
			LOGGER.log(Level.INFO, "Weights loaded from " + path);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	@Override
	public void releaseCLMem() {
		Layer currentLayer = inputLayer;
		while (currentLayer != null) {
			currentLayer.releaseCLMem();
			currentLayer = currentLayer.getNextLayer();
		}
	}

}
