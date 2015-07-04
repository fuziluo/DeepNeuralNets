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
import com.changjinxiong.deepneuralnets.nn.PoolingLayer;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer.PoolingType;
import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;

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
		int inputSize = 224;
		int[][] para = {{numOfInputFeatureMaps, 0, 0, 0}, 
						{numOfOutputFeatureMaps, filterSize, filterSize, stride},
						{2, 2, 2},
						{256, 5, 5, 1},
						{2, 2, 2},
						{384, 3, 3, 1},
						{384, 3, 3, 1},
						{256, 3, 3, 1},
						{4096},
						{4096},
						{1000}
				};
		boolean addBias = true;
		boolean useOpenCL = true;
		boolean padding = true;
		int batchSize = 256;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, padding, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		Layer l5 = l4.getNextLayer();
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
}
