package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.ConvolutionalNeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.FeatureMapLayer;
import com.changjinxiong.deepneuralnets.nn.ConvolutionalLayer;
import com.changjinxiong.deepneuralnets.nn.FullyConnectedLayer;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer.PoolingType;
import com.changjinxiong.deepneuralnets.test.CIFAR10DataProvider.DatasetType;

public class TestPooling {

	@Test
	public void testBackpropagation() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {2, 2}, {10}};
		int[][] para = {{3, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {2, 2}, {10}};
		boolean addBias = true;
		boolean useOpenCL = false;
		int batchSize = 100;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		PoolingLayer l4 = (PoolingLayer) l3.getNextLayer();
//		l4.setPoolingType(PoolingType.AVER);
		
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		cnn.setInputShape(new int[] {32, 32});
		CIFAR10DataProvider tp = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);

		
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		cnn.fordwardPass(tin);
		cnn.backPropagation(tout, 0);
		int i = 0;
		Layer l = l3;
		float g1 = l.getGradients()[i];
//		System.out.println(Arrays.toString(l.getGradients()));
//		System.out.println(Arrays.toString(l4.getPrevErrors()));
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.005f;
		weights[i] = (float) (w - e);
		l.setWeight(weights);
		cnn.fordwardPass(tin);
		float c1 = cnn.getCost(tout, 0);
//		float[] a1 = l.getActivations();

		weights[i] = (float) (w + e);
		l.setWeight(weights);

		cnn.fordwardPass(tin);
		float c2 = cnn.getCost(tout, 0);
		double g2 = (c2 - c1)/(2 * e);
//		float[] a2 = l.getActivations();
//		assertArrayEquals("!!",a1,a2, 0.0001f);

		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 0.0015);
	}
	
	@Test
	public void testBackpropagationCL() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {2, 2}, {10}};
		int[][] para = {{3, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {2, 2}, {10}};
		boolean addBias = true;
		boolean useOpenCL = true;
		int batchSize = 1;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		PoolingLayer l4 = (PoolingLayer) l3.getNextLayer();
		l4.setPoolingType(PoolingType.MAX);
		
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		cnn.setInputShape(new int[] {32, 32});
		CIFAR10DataProvider tp = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);

		
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		cnn.fordwardPass(tin);
		cnn.backPropagation(tout, 0);
		int i = 0;
		Layer l = l3;
		float g1 = l.getGradients()[i];
//		System.out.println(Arrays.toString(l.getGradients()));
//		System.out.println(Arrays.toString(l4.getPrevErrors()));
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.00125f;
		weights[i] = (float) (w - e);
		l.setWeight(weights);
		cnn.fordwardPass(tin);
		float c1 = cnn.getCost(tout, 0);
//		float[] a1 = l.getActivations();

		weights[i] = (float) (w + e);
		l.setWeight(weights);

		cnn.fordwardPass(tin);
		float c2 = cnn.getCost(tout, 0);
		double g2 = (c2 - c1)/(2 * e);
//		float[] a2 = l.getActivations();
//		assertArrayEquals("!!",a1,a2, 0.0001f);

		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 0.0015);
	}

	@Test
	public void testForwardPass() {
		boolean addBias = false;
		boolean useOpenCL = false;
		int numOfFeatureMaps = 2;
		int poolHeight = 2;
		int poolWidth = 2;
		FeatureMapLayer inputLayer = new ConvolutionalLayer(numOfFeatureMaps, 0, 0, 0, null, null, addBias, useOpenCL);
		FeatureMapLayer poolingLayer = new PoolingLayer(poolHeight, poolWidth, inputLayer, null, useOpenCL);
		inputLayer.setNextLayer(poolingLayer);
//		Layer outLayer = new FullyConnectedLayer(10, poolingLayer, null, false, useOpenCL);
		//test max pooling
		int[] inputShape = {6, 6};
		float[] inputs = {
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,

				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,				

				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,

				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,				
		};
		float[] outputs = {
				9, 4, 5,
				2, 4, 5,
				2, 8, 5,

				9, 4, 5,
				2, 4, 5,
				2, 8, 5,

				9, 4, 5,
				2, 4, 5,
				2, 8, 5,

				9, 4, 5,
				2, 4, 5,
				2, 8, 5,
		};
		inputLayer.setInputShape(inputShape);
		inputLayer.setInputs(inputs);
		
		poolingLayer.forwardPass();
		float[] results = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs,results, 0);

		int[] inputShape1 = {5, 5};
		float[] inputs1 = {
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,

				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
		};
		float[] outputs1 = {
				9, 4,
				2, 4,

				9, 4,
				2, 4,
		};
		inputLayer.setInputShape(inputShape1);
		inputLayer.setInputs(inputs1);
		
		poolingLayer.forwardPass();
		float[] results1 = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs1,results1, 0);		
	}
	@Test
	public void testForwardPassAverPooling() {
		boolean addBias = false;
		boolean useOpenCL = false;
		int numOfFeatureMaps = 2;
		int poolHeight = 2;
		int poolWidth = 2;
		FeatureMapLayer inputLayer = new ConvolutionalLayer(numOfFeatureMaps, 0, 0, 0, null, null, addBias, useOpenCL);
		PoolingLayer poolingLayer = new PoolingLayer(poolHeight, poolWidth, inputLayer, null, useOpenCL);
		poolingLayer.setPoolingType(PoolingType.AVER);
		inputLayer.setNextLayer(poolingLayer);
//		Layer outLayer = new FullyConnectedLayer(10, poolingLayer, null, false, useOpenCL);
		//test average pooling
		int[] inputShape = {6, 6};
		float[] inputs = {
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,

				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,				
		};
		float[] outputs = {
				3.25f, 3.5f, 2.75f,
				1.5f, 3.5f, 3.25f,
				1.5f, 4.5f, 3.25f,

				3.25f, 3.5f, 2.75f,
				1.5f, 3.5f, 3.25f,
				1.5f, 4.5f, 3.25f,
		};
		inputLayer.setInputShape(inputShape);
		inputLayer.setInputs(inputs);
		
		poolingLayer.forwardPass();
		float[] results = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs,results, 0);

		int[] inputShape1 = {5, 5};
		float[] inputs1 = {
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,

				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,

				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,

				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
		};
		float[] outputs1 = {
				3.25f, 3.5f,
				1.5f, 3.5f, 

				3.25f, 3.5f,
				1.5f, 3.5f, 

				3.25f, 3.5f,
				1.5f, 3.5f, 

				3.25f, 3.5f,
				1.5f, 3.5f, 
		};
		inputLayer.setInputShape(inputShape1);
		inputLayer.setInputs(inputs1);
		
		poolingLayer.forwardPass();
		float[] results1 = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs1,results1, 0);	
	
	}

	@Test
	public void testForwardPassCL() {
		boolean addBias = false;
		boolean useOpenCL = true;
		int numOfFeatureMaps = 2;
		int poolHeight = 2;
		int poolWidth = 2;
		FeatureMapLayer inputLayer = new ConvolutionalLayer(numOfFeatureMaps, 0, 0, 0, null, null, addBias, useOpenCL);
		FeatureMapLayer poolingLayer = new PoolingLayer(poolHeight, poolWidth, inputLayer, null, useOpenCL);
		inputLayer.setNextLayer(poolingLayer);
	//	Layer outLayer = new FullyConnectedLayer(10, poolingLayer, null, false, useOpenCL);
		//test max pooling
		int[] inputShape = {6, 6};
		float[] inputs = {
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,
	
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,				
	
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,
	
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,				
		};
		float[] outputs = {
				9, 4, 5,
				2, 4, 5,
				2, 8, 5,
	
				9, 4, 5,
				2, 4, 5,
				2, 8, 5,
	
				9, 4, 5,
				2, 4, 5,
				2, 8, 5,
	
				9, 4, 5,
				2, 4, 5,
				2, 8, 5,
		};
		inputLayer.setInputShape(inputShape);
		inputLayer.setInputs(inputs);
		
		poolingLayer.forwardPass();
		float[] results = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs,results, 0);
	
		int[] inputShape1 = {5, 5};
		float[] inputs1 = {
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
	
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
		};
		float[] outputs1 = {
				9, 4,
				2, 4,
	
				9, 4,
				2, 4,
		};
		inputLayer.setInputShape(inputShape1);
		inputLayer.setInputs(inputs1);
		
		poolingLayer.forwardPass();
		float[] results1 = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs1,results1, 0);		
	}
	@Test
	public void testForwardPassAverPoolingCL() {
		boolean addBias = false;
		boolean useOpenCL = true;
		int numOfFeatureMaps = 2;
		int poolHeight = 2;
		int poolWidth = 2;
		FeatureMapLayer inputLayer = new ConvolutionalLayer(numOfFeatureMaps, 0, 0, 0, null, null, addBias, useOpenCL);
		PoolingLayer poolingLayer = new PoolingLayer(poolHeight, poolWidth, inputLayer, null, useOpenCL);
		poolingLayer.setPoolingType(PoolingType.AVER);
		inputLayer.setNextLayer(poolingLayer);
	//	Layer outLayer = new FullyConnectedLayer(10, poolingLayer, null, false, useOpenCL);
		//test average pooling
		int[] inputShape = {6, 6};
		float[] inputs = {
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,
	
				1, 2, 3, 4, 5, 1,
				1, 9, 3, 4, 3, 2,
				1, 2, 3, 4, 5, 1,
				1, 2, 3, 4, 5, 2,
				1, 2, 3, 8, 5, 1,
				1, 2, 3, 4, 5, 2,				
		};
		float[] outputs = {
				3.25f, 3.5f, 2.75f,
				1.5f, 3.5f, 3.25f,
				1.5f, 4.5f, 3.25f,
	
				3.25f, 3.5f, 2.75f,
				1.5f, 3.5f, 3.25f,
				1.5f, 4.5f, 3.25f,
		};
		inputLayer.setInputShape(inputShape);
		inputLayer.setInputs(inputs);
		
		poolingLayer.forwardPass();
		float[] results = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs,results, 0);
	
		int[] inputShape1 = {5, 5};
		float[] inputs1 = {
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
	
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
	
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
	
				1, 2, 3, 4, 5,
				1, 9, 3, 4, 3,
				1, 2, 3, 4, 5,
				1, 2, 3, 4, 5,
				1, 2, 3, 8, 5,
		};
		float[] outputs1 = {
				3.25f, 3.5f,
				1.5f, 3.5f, 
	
				3.25f, 3.5f,
				1.5f, 3.5f, 
	
				3.25f, 3.5f,
				1.5f, 3.5f, 
	
				3.25f, 3.5f,
				1.5f, 3.5f, 
		};
		inputLayer.setInputShape(inputShape1);
		inputLayer.setInputs(inputs1);
		
		poolingLayer.forwardPass();
		float[] results1 = poolingLayer.getActivations();
		System.out.println(Arrays.toString(results));
		assertArrayEquals("!!",outputs1,results1, 0);	
	
	}
}