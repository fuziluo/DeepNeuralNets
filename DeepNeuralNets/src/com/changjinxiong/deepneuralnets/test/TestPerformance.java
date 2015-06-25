package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.ConvolutionalNeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.FeatureMapLayer;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;

public class TestPerformance {

	@Test
	public void testMLPForward() {
		int inputLayerSize = 800000;
		int outputLayerSize = 8;
		int batchSize = 128;
		boolean useOpenCL = true;
		boolean addBias = false;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{inputLayerSize, outputLayerSize}, addBias, useOpenCL);
		float[] inputSamples = new float[(inputLayerSize + (addBias ? 1 : 0)) * batchSize];
		mlp.getInputLayer().setInputs(inputSamples); //provide input data
		mlp.getOutputLayer().forwardPass();
	}
	
	@Test
	public void testCNNTiming() {
		int numOfInputFeatureMaps = 3;
		int numOfOutputFeatureMaps = 128;
		int filterSize = 5;
		int stride = 1;
		int inputSize = 128;
		int[][] para = {{numOfInputFeatureMaps, 0, 0, 0}, 
						{numOfOutputFeatureMaps, filterSize, filterSize, stride},
						{2, 2},
						{numOfOutputFeatureMaps, filterSize, filterSize, stride},
						{2, 2},
						{500},
						{10}
				};
		boolean addBias = true;
		boolean useOpenCL = true;
		int batchSize = 32;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		Layer l5 = l4.getNextLayer();
		cnn.setInputShape(new int[] {inputSize, inputSize});		
		float[] testInput = new float[batchSize * numOfInputFeatureMaps * inputSize * inputSize];
		float[] testLabel = new float[batchSize * 10];
		cnn.fordwardPass(testInput);
		System.out.println("**********Timing starting************");
		long t = System.currentTimeMillis();
		cnn.fordwardPass(testInput);
		System.out.println("forward "+(System.currentTimeMillis() - t));
		t = System.currentTimeMillis();
		cnn.backPropagation(testLabel, 0);
		System.out.println("back "+(System.currentTimeMillis() - t));

	}

	
	@Test
	public void testCNNTiming1() {
		boolean useOpenCL = true;
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
		int[][] cnnLayers = new int[][] {{1, 0, 0 ,0}, {20, 5, 5, 1},{2, 2}, {50, 5, 5, 1},{2, 2}, {500}, {10}};
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(cnnLayers, addBias, useOpenCL); 
		cnn.setInputShape(new int[] {28, 28});
		Layer l1 = cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		Layer l5 = l4.getNextLayer();
		Layer l6 = l5.getNextLayer();
		Layer l7 = l6.getNextLayer();
		l2.setActivationType(ActivationType.TANH);
		l4.setActivationType(ActivationType.TANH);
		l6.setActivationType(ActivationType.TANH);
		l7.setActivationType(ActivationType.TANH);

	
		cnn.train(trainingSet, costType, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate, epoch);


		
	}
}
