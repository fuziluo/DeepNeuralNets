package com.changjinxiong.deepneuralnets.play;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.ConvolutionalLayer;
import com.changjinxiong.deepneuralnets.nn.ConvolutionalNeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.FeatureMapLayer;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer.PoolingType;
import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.opencl.OpenCL;
import com.changjinxiong.deepneuralnets.test.MnistDataProvider;

public class TestPerformance {

	@Test
	public void testMLPTiming() {
//		int inputLayerSize = 13*13;
//		int hiddenLayerSize = 256*13*13;
//		int outputLayerSize = 4096;
//		int batchSize = 256;
		int inputLayerSize = 4096;
		int hiddenLayerSize = 4096;
		int outputLayerSize = 1000;
		int batchSize = 256;
		boolean useOpenCL = true;
		boolean addBias = true;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{inputLayerSize, hiddenLayerSize, outputLayerSize}, addBias, useOpenCL);
		float[] inputSamples = new float[inputLayerSize * batchSize];
		float[] labels = new float[outputLayerSize * batchSize];
//		mlp.getInputLayer().setInputs(inputSamples); //provide input data
		mlp.forwardPass(inputSamples);
		long t = System.currentTimeMillis();
		mlp.forwardPass(inputSamples);
		long t1 = System.currentTimeMillis();
		System.out.println("forward "+(t1 - t));
		mlp.backPropagation(labels, 1);
		long t2 = System.currentTimeMillis();
		System.out.println("back "+(t2 - t1));
		
	}
	
	@Test
	public void testCNNTiming() {
		int numOfInputFeatureMaps = 3;
		int numOfOutputFeatureMaps = 96;
		int filterSize = 11;
		int stride = 4;
		int inputSize = 227;
		int[][] para = {{3, 0, 0, 0}, 
						{96, 11, 11, 4},
						{3, 3, 2},
						{256, 5, 5, 1},
						{3, 3, 2},
						{384, 3, 3, 1},
						{384, 3, 3, 1},
						{256, 3, 3, 1},
						{3, 3, 2},
						{4096},
						{4096},
						{1000}
				};
		boolean addBias = true;
		boolean useOpenCL = true;
		boolean padding = false;
		int batchSize = 256;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, padding, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		ConvolutionalLayer l4 = (ConvolutionalLayer) l3.getNextLayer();
		Layer l5 = l4.getNextLayer();
		ConvolutionalLayer l6 = (ConvolutionalLayer) l5.getNextLayer();
		ConvolutionalLayer l7 = (ConvolutionalLayer) l6.getNextLayer();
		ConvolutionalLayer l8 = (ConvolutionalLayer) l7.getNextLayer();
		l4.setPadding(true);
		l6.setPadding(true);
		l7.setPadding(true);
		l8.setPadding(true);
		cnn.setInputShape(new int[] {inputSize, inputSize});		
		float[] testInput = new float[batchSize * numOfInputFeatureMaps * inputSize * inputSize];
		float[] testLabel = new float[batchSize * 1000];
		cnn.forwardPass(testInput);
		cnn.releaseCLMem();
		System.out.println("**********Timing starting************");
		long t = System.currentTimeMillis();
		cnn.forwardPass(testInput);
		System.out.println("forward "+(System.currentTimeMillis() - t));
		t = System.currentTimeMillis();
		cnn.backPropagation(testLabel, 0);
		System.out.println("back "+(System.currentTimeMillis() - t));

	}

	
	@Test
	public void testCNNTiming1() {
		boolean useOpenCL = true;
		boolean padding = false;
		boolean addBias = true;
		int batchSize = 6000;
		MnistDataProvider trainingSet = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
//		MnistDataProvider testSet = new MnistDataProvider("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", batchSize, false);
		int costType = 1; //cross entropy
		float baselearningRate = 0.01f;
		float momentum = 0.9f;
		float weightDecay = 0.0005f;
		int lrChangeCycle = 0;//5 * trainingSet.getDatasetSize()/trainingSet.getBatchSize();
		float lrChangeRate = 0.33f;
		int epoch = 1;
		int[][] cnnLayers = new int[][] {{1, 0, 0 ,0}, {20, 5, 5, 1},{2, 2, 2}, {50, 5, 5, 1},{2, 2, 2}, {500}, {10}};
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(cnnLayers, addBias, padding, useOpenCL); 
		cnn.setInputShape(new int[] {28, 28});
		Layer l1 = cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		PoolingLayer l3 = (PoolingLayer) l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		PoolingLayer l5 = (PoolingLayer) l4.getNextLayer();
		Layer l6 = l5.getNextLayer();
		Layer l7 = l6.getNextLayer();
		l2.setActivationType(ActivationType.TANH);
		l4.setActivationType(ActivationType.TANH);
		l6.setActivationType(ActivationType.TANH);
		l7.setActivationType(ActivationType.TANH);
		l3.setPoolingType(PoolingType.AVER);
		l5.setPoolingType(PoolingType.AVER);
	
		cnn.train(trainingSet, costType, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate, epoch);

	}
	
	
	@Test
	public void testWorkGroupSizeBackGradients() {
		int numOfInputFeatureMaps = 128;
		int numOfOutputFeatureMaps = 128;
		int filterSize = 5;
		int stride = 1;
		int inputSize = 50;
		int labelSize = 100;
		int[][] para = {{numOfInputFeatureMaps, 0, 0, 0}, 
						{numOfOutputFeatureMaps, filterSize, filterSize, stride},
						{labelSize}
				};
		boolean addBias = true;
		boolean useOpenCL = true;
		boolean padding = true;
		int batchSize = 128;
		
		
		for (batchSize = 128; batchSize >= 1; batchSize /= 4) {
//		for (numOfInputFeatureMaps = 32; numOfInputFeatureMaps >= 8; numOfInputFeatureMaps /= 4) {
//		for (numOfOutputFeatureMaps = 32; numOfOutputFeatureMaps >= 8; numOfOutputFeatureMaps /= 4) {
			ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, padding, useOpenCL);
			Layer l1 = cnn.getInputLayer();
			ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
			cnn.setInputShape(new int[] {inputSize, inputSize});		
			float[] testInput = new float[batchSize * numOfInputFeatureMaps * inputSize * inputSize];
			float[] testLabel = new float[batchSize * labelSize];
			System.out.println("Network: "+Arrays.deepToString(para));
			System.out.println("batch size: "+batchSize);
			cnn.forwardPass(testInput);
			cnn.releaseCLMem();
			
			int maxSize = 256;
			int[] grouSize = new int[] {
					4, 4, 4,
					1, 1, maxSize,
					16, 8, 2,
			};
			for (grouSize[3] = 2; grouSize[3] <= maxSize; grouSize[3] *= 2) {
				for (grouSize[4] = 2; grouSize[4] <= maxSize / grouSize[3]; grouSize[4] *= 2) {
					grouSize[5] = maxSize / grouSize[3] / grouSize[4];
						System.out.print(grouSize[3] +" "+ grouSize[4] + " " + grouSize[5] + " ");
						OpenCL.setTestGrpSize(grouSize);
						l2.setRegenerateKernels();
						try{
						cnn.forwardPass(testInput);
						
						
						cnn.calCostErr(testLabel, 0);
						cnn.getOutputLayer().backpropagation();
						l2.backpropagation();
						cnn.releaseCLMem();
						}catch(Exception e){
							System.out.println("#");
//							OpenCL.releaseAll();
						}
				}
			}
			OpenCL.releaseAll();
		
		}
	}
	
	@Test
	public void testWorkGroupSizeBackPrevErr() {
		int numOfInputFeatureMaps = 128;
		int numOfOutputFeatureMaps = 128;
		int filterSize = 5;
		int stride = 1;
		int inputSize = 50;
		int labelSize = 100;
		int[][] para = {{numOfInputFeatureMaps, 0, 0, 0}, 
						{numOfInputFeatureMaps, filterSize, filterSize, stride},
						{numOfOutputFeatureMaps, filterSize, filterSize, stride},
						{labelSize}
				};
		boolean addBias = true;
		boolean useOpenCL = true;
		boolean padding = true;
		int batchSize = 128;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, padding, useOpenCL);
		Layer l1 = cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		ConvolutionalLayer l3 = (ConvolutionalLayer) l2.getNextLayer();
		cnn.setInputShape(new int[] {inputSize, inputSize});		
		float[] testInput = new float[batchSize * numOfInputFeatureMaps * inputSize * inputSize];
		float[] testLabel = new float[batchSize * labelSize];
		cnn.forwardPass(testInput);
		cnn.releaseCLMem();

		System.out.println("Network: "+Arrays.deepToString(para));
		System.out.println("batch size: "+batchSize);
		
		int maxSize = 256;
		int[] grouSize = new int[] {
				4, 4, 4,
				1, 16, 4,
				1, 1, maxSize,
		};
		for (grouSize[6] = 1; grouSize[6] <= maxSize; grouSize[6] *= 2) {
			for (grouSize[7] = 1; grouSize[7] <= maxSize / grouSize[6]; grouSize[7] *= 2) {
				grouSize[8] = maxSize / grouSize[6] / grouSize[7];
					System.out.print(grouSize[6] +" "+ grouSize[7] + " " + grouSize[8] + " ");
					OpenCL.setTestGrpSize(grouSize);
					l3.setRegenerateKernels();
					cnn.forwardPass(testInput);
					cnn.calCostErr(testLabel, 0);
					cnn.getOutputLayer().backpropagation();
					l3.backpropagation();
					cnn.releaseCLMem();

			}
		}
	}
}
