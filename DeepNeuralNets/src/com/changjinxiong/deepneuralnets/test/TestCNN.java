package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.nio.file.Paths;
import java.util.Arrays;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.ConvolutionalLayer;
import com.changjinxiong.deepneuralnets.nn.ConvolutionalNeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.FeatureMapLayer;
import com.changjinxiong.deepneuralnets.nn.FullyConnectedLayer;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer.PoolingType;
import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.opencl.OpenCL;
import com.changjinxiong.deepneuralnets.test.CIFAR10DataProvider.DatasetType;

public class TestCNN {

	@Test
	public void testConvLayerForward() {
		boolean useOpenCL = true;
		ConvolutionalLayer cl1 = new ConvolutionalLayer(2, 0, 0, 0, null, null, false, useOpenCL);
		ConvolutionalLayer cl2 = new ConvolutionalLayer(2, 2, 2, 1, cl1, null, true, useOpenCL);
		cl1.setNextLayer(cl2);
		int[] inputShape = {3, 3};
		cl1.setInputShape(inputShape);
		float[] inputs = {	0.1f, 0.2f, 0.3f,
							0.4f, 0.5f, 0.6f,
							0.7f, 0.8f, 0.9f,
							1.1f, 1.2f, 1.3f,
							1.4f, 1.5f, 1.6f,
							1.7f, 1.8f, 1.9f,
							2.1f, 2.2f, 2.3f,
							2.4f, 2.5f, 2.6f,
							2.7f, 2.8f, 2.9f,
							3.1f, 3.2f, 3.3f,
							3.4f, 3.5f, 3.6f,
							3.7f, 3.8f, 3.9f,
						};
		cl1.setInputs(inputs);
		float[] weights = {	0.1f, 0.2f, 
							0.3f, 0.4f, 0.5f,
							0.6f, 0.7f,
							0.8f, 0.9f, 1,
							
							1.1f, 1.2f, 
							1.3f, 1.4f, 1.5f,
							1.6f, 1.7f,
							1.8f, 1.9f, 2,
							};
		cl2.setWeight(weights);
		cl2.setActivationType(ActivationType.SIGMOID);
		cl2.forwardPass();
		float[] act = cl2.getActivations();
		float[] actCorrect = {	
				5.8400f,    6.2400f,
			    7.0400f,    7.4400f,
			    14.2400f,   15.4400f,
			    17.8400f,   19.0400f,
			    
			    13.8400f,   14.2400f,
			    15.0400f,   15.4400f,
			    38.2400f,   39.4400f,
			    41.8400f,   43.0400f,

		};
		for (int i = 0; i < actCorrect.length; i++) {
			actCorrect[i] = (float) (1 / (1 + Math.exp(-actCorrect[i])));
		}
		System.out.println(Arrays.toString(actCorrect));
		System.out.println(Arrays.toString(act));
		assertArrayEquals("!!",actCorrect,act, 0.0001f);
//		OpenCL.releaseAll();

	}

	@Test
	public void testConvLayerForwardIrregularInputShape() {
		boolean useOpenCL = true;
		ConvolutionalLayer cl1 = new ConvolutionalLayer(2, 0, 0, 0, null, null, false, useOpenCL);
		ConvolutionalLayer cl2 = new ConvolutionalLayer(2, 2, 2, 2, cl1, null, true, useOpenCL);
		cl1.setNextLayer(cl2);
		cl2.setActivationType(ActivationType.SIGMOID);
		int[] inputShape = {3, 5};
		cl1.setInputShape(inputShape);
		float[] inputs = {	0.1f, 0.2f, 0.3f, 0.2f, 0.3f,
							0.4f, 0.5f, 0.6f, 0.5f, 0.6f,
							0.7f, 0.8f, 0.9f, 0.8f, 0.9f,
							1.1f, 1.2f, 1.3f, 1.2f, 1.3f,
							1.4f, 1.5f, 1.6f, 1.5f, 1.6f,
							1.7f, 1.8f, 1.9f, 1.8f, 1.9f,
							2.1f, 2.2f, 2.3f, 2.2f, 2.3f,
							2.4f, 2.5f, 2.6f, 2.5f, 2.6f,
							2.7f, 2.8f, 2.9f, 2.8f, 2.9f,
							3.1f, 3.2f, 3.3f, 3.2f, 3.3f,
							3.4f, 3.5f, 3.6f, 3.5f, 3.6f,
							3.7f, 3.8f, 3.9f, 3.8f, 3.9f,
						};
		cl1.setInputs(inputs);
		float[] weights = {	0.1f, 0.2f, 
							0.3f, 0.4f, 0.5f,
							0.1f, 0.2f, 
							0.3f, 0.4f, 0.5f,
							0.6f, 0.7f,
							0.8f, 0.9f, 1,
							0.6f, 0.7f,
							0.8f, 0.9f, 1,
							};
		cl2.setWeight(weights);
		cl2.forwardPass();
		float[] act = cl2.getActivations();
		float[] actCorrect = {	2.74f, 2.9f, 
								6.94f, 7.5f, 
								6.74f, 6.9f, 
								18.94f, 19.5f, 
								};
		for (int i = 0; i < actCorrect.length; i++) {
			actCorrect[i] = (float) (1 / (1 + Math.exp(-actCorrect[i])));
		}
		System.out.println(Arrays.toString(act));
		assertArrayEquals("!!",act,actCorrect, 0.0001f);
//		OpenCL.releaseAll();
	}
	
	@Test
	public void testConvLayerForwardPadding() {
		boolean useOpenCL = true;
		boolean padding = true;
		ConvolutionalLayer cl1 = new ConvolutionalLayer(2, 0, 0, 0, null, null, false, useOpenCL);
		ConvolutionalLayer cl2 = new ConvolutionalLayer(2, 2, 2, 1, cl1, null, true, useOpenCL);
		cl1.setNextLayer(cl2);
		cl2.setPadding(padding);
		int[] inputShape = {3, 3};
		cl1.setInputShape(inputShape);
		float[] inputs = {	0.1f, 0.2f, 0.3f,
							0.4f, 0.5f, 0.6f,
							0.7f, 0.8f, 0.9f,
							1.1f, 1.2f, 1.3f,
							1.4f, 1.5f, 1.6f,
							1.7f, 1.8f, 1.9f,
							
							0.1f, 0.2f, 0.3f,
							0.4f, 0.5f, 0.6f,
							0.7f, 0.8f, 0.9f,
							1.1f, 1.2f, 1.3f,
							1.4f, 1.5f, 1.6f,
							1.7f, 1.8f, 1.9f,
						};
		cl1.setInputs(inputs);
		float[] weights = {	0.1f, 0.2f, 
							0.3f, 0.4f, 0.5f,
							0.1f, 0.2f, 
							0.3f, 0.4f, 0.5f,
							0.6f, 0.7f,
							0.8f, 0.9f, 1,
							0.6f, 0.7f,
							0.8f, 0.9f, 1,
							};
		cl2.setWeight(weights);
		cl2.setActivationType(ActivationType.SIGMOID);
		cl2.forwardPass();
		float[] act = cl2.getActivations();
		float[] actCorrect = {	    2.7400f,    2.9400f,    1.8200f,
								    3.3400f,    3.5400f,    2.0600f,
								    1.7600f,    1.8200f,    1.2800f, 
								    6.9400f,    7.5400f,    4.7200f,
								    8.7400f,    9.3400f,    5.5600f,
								    5.2600f,    5.5200f,    3.6800f,
								    
								    2.7400f,    2.9400f,    1.8200f,
								    3.3400f,    3.5400f,    2.0600f,
								    1.7600f,    1.8200f,    1.2800f, 
								    6.9400f,    7.5400f,    4.7200f,
								    8.7400f,    9.3400f,    5.5600f,
								    5.2600f,    5.5200f,    3.6800f 
							};
		for (int i = 0; i < actCorrect.length; i++) {
			actCorrect[i] = (float) (1 / (1 + Math.exp(-actCorrect[i])));
		}
		System.out.println(Arrays.toString(actCorrect));
		System.out.println(Arrays.toString(act));
		assertArrayEquals("!!",actCorrect,act, 0.0001f);
//		OpenCL.releaseAll();
	}
	
	@Test
	public void testConstructionCNN() {
		int[][] para = {{2, 0, 0, 0}, {4, 3, 3, 1}, {3, 2, 2, 2}, {4}, {5}};
		boolean addBias = true;
		boolean useOpenCL = false;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		Layer l5 = l4.getNextLayer();
		l1.setInputShape(new int[] {4, 4});

		assertEquals(l2.getWeight().length, 80, 0);
		assertEquals(l3.getWeight().length, 60, 0);
		assertNull(l4.getWeight());
		assertEquals(l5.getWeight().length, 25, 0);
		
		float[] testInput = {	0.1f, 0.2f, 0.1f, 0.2f,
								0.3f, 0.4f, 0.3f, 0.4f,
								0.1f, 0.2f, 0.1f, 0.2f,
								0.3f, 0.4f, 0.3f, 0.4f,
								
								0.5f, 0.6f, 0.5f, 0.6f,
								0.7f, 0.8f, 0.7f, 0.8f,
								0.5f, 0.6f, 0.5f, 0.6f,
								0.7f, 0.8f, 0.7f, 0.8f,
								
								1.1f, 1.2f, 1.1f, 1.2f,
								1.3f, 1.4f, 1.3f, 1.4f,
								1.1f, 1.2f, 1.1f, 1.2f,
								1.3f, 1.4f, 1.3f, 1.4f,
								
								1.5f, 1.6f, 1.5f, 1.6f,
								1.7f, 1.8f, 1.7f, 1.8f,
								1.5f, 1.6f, 1.5f, 1.6f,
								1.7f, 1.8f, 1.7f, 1.8f,
		};
		int[] inputShape = {4, 4};
		cnn.setInputShape(inputShape);
		cnn.forwardPass(testInput);
		assertEquals(l4.getWeight().length, 16, 0);
		float[] act = l5.getActivations();
//		System.out.println(Arrays.toString(act));
//		OpenCL.releaseAll();
	}

	@Test
	public void testForwardPassCNN() {
		int[][] para = {{2, 0, 0, 0}, {2, 3, 3, 1}, {4}};
		boolean addBias = true;
		boolean useOpenCL = true;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		cnn.setInputShape(new int[] {4, 4});
		assertEquals(l2.getWeight().length, 40, 0);
//		assertNull(l3.getWeight());
		l2.setActivationType(ActivationType.SIGMOID);
		l3.setActivationType(ActivationType.SIGMOID);
		
		float[] testInput = {	0.1f, 0.2f, 0.1f, 0.2f,
								0.3f, 0.4f, 0.3f, 0.4f,
								0.1f, 0.2f, 0.1f, 0.2f,
								0.3f, 0.4f, 0.3f, 0.4f,
								
								0.5f, 0.6f, 0.5f, 0.6f,
								0.7f, 0.8f, 0.7f, 0.8f,
								0.5f, 0.6f, 0.5f, 0.6f,
								0.7f, 0.8f, 0.7f, 0.8f,
								
								1.1f, 1.2f, 1.1f, 1.2f,
								1.3f, 1.4f, 1.3f, 1.4f,
								1.1f, 1.2f, 1.1f, 1.2f,
								1.3f, 1.4f, 1.3f, 1.4f,
								
								1.5f, 1.6f, 1.5f, 1.6f,
								1.7f, 1.8f, 1.7f, 1.8f,
								1.5f, 1.6f, 1.5f, 1.6f,
								1.7f, 1.8f, 1.7f, 1.8f,
		};
		int[] inputShape = {4, 4};
		cnn.setInputShape(inputShape);
		
		float[] w2 = {	0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, 1.0f,
						0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, 1.0f,
						1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f, 1.7f, -1.8f, 1.9f, 2.0f,
						1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f, 1.7f, -1.8f, 1.9f, 2.0f,
		};
		l2.setWeight(w2);
		float[] w3 = {	0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f,
						1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f, 1.7f, -1.8f, 1.9f,
						2.1f, -2.2f, 2.3f, -2.4f, 2.5f, -2.6f, 2.7f, -2.8f, 2.9f,
						3.1f, -3.2f, 3.3f, -3.4f, 3.5f, -3.6f, 3.7f, -3.8f, 3.9f,
		};
		l3.setWeight(w3);
		
		float[] actCorrect = {	0.6226324f, 0.8089633f, 0.91574115f, 0.96539f, 
								0.6229384f, 0.81499475f, 0.9215472f, 0.96906114f
		};
		
		cnn.forwardPass(testInput);
		assertEquals(l3.getWeight().length, 36, 0);
		float[] act2 = l2.getActivations();
		float[] act3 = l3.getActivations();
//		System.out.println(Arrays.toString(act2));
		System.out.println(Arrays.toString(act3));

		assertArrayEquals("!!",act3,actCorrect, 0.0001f);
//		OpenCL.releaseAll();
	}

	@Test
	public void gradientCheck() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
		int[][] para = {{3, 0, 0, 0}, {3, 3, 3, 1}, {4, 3, 3, 1}, {4, 3, 3, 1}, {10}};
		boolean addBias = true;
		boolean useOpenCL = true;
		int costType = 0;
		int batchSize = 100;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		ConvolutionalLayer l3 = (ConvolutionalLayer) l2.getNextLayer();
		ConvolutionalLayer l4 = (ConvolutionalLayer) l3.getNextLayer();
		l2.initializeWeights(0.0125f, 0);
		l3.initializeWeights(0.25f, 0);
		l4.initializeWeights(0.25f, 0);
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		cnn.setInputShape(new int[] {32, 32});
		String path = Paths.get(System.getProperty("user.dir"), "..", "..", "..","datasets", "CIFAR", "cifar-10-batches-bin").toString();
		CIFAR10DataProvider tp = new CIFAR10DataProvider(path, batchSize, DatasetType.TRAINING_ALL, false);


		
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		cnn.forwardPass(tin);
		
		cnn.backPropagation(tout, costType);
		int i = 10;
		Layer l = l2;
		float g1 = l.getGradients()[i];
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.001f;
		weights[i] = (float) (w - e);
		l.setWeight(weights);
		cnn.forwardPass(tin);
		cnn.calCostErr(tout, costType);
		float c1 = cnn.getCost();
//		float[] a1 = l.getActivations();

		weights[i] = (float) (w + e);
		l.setWeight(weights);

		cnn.forwardPass(tin);
		cnn.calCostErr(tout, costType);
		float c2 = cnn.getCost();
		double g2 = (c2 - c1)/(2 * e);
//		float[] a2 = l.getActivations();
//		assertArrayEquals("!!",a1,a2, 0.0001f);

		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 0.0015);
//		assertEquals(0, (g1-g2)/(Math.abs(g1)+Math.abs(g2)), 0.000015);
		cnn.releaseCLMem();
//		OpenCL.releaseAll();
	}
	
	@Test
	public void gradientCheckReLU() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
//		int[][] para = {{3, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
		int[][] para = new int[][] {	
				{3, 0, 0 ,0}, 
				{32, 5, 5, 1},
				{3, 3, 2}, 
				{32, 5, 5, 1},
				{3, 3, 2}, 
				{64, 5, 5, 1}, 
				{3, 3, 2}, 
				{64},
				{10}
				};
		boolean addBias = true;
		boolean useOpenCL = true;
		int costType = 0;
		int batchSize = 100;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, true, useOpenCL);
//		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
//		Layer l2 = l1.getNextLayer();
//		Layer l3 = l2.getNextLayer();
//		Layer l4 = l3.getNextLayer();
//		l2.setActivationType(ActivationType.RELU);
//		l3.setActivationType(ActivationType.RELU);
//		l4.setActivationType(ActivationType.NONE);

		Layer l1 = cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		PoolingLayer l3 = (PoolingLayer) l2.getNextLayer();
		ConvolutionalLayer l4 = (ConvolutionalLayer) l3.getNextLayer();
		PoolingLayer l5 = (PoolingLayer) l4.getNextLayer();
		ConvolutionalLayer l6 = (ConvolutionalLayer) l5.getNextLayer();
		PoolingLayer l7 = (PoolingLayer) l6.getNextLayer();
		FullyConnectedLayer l8 = (FullyConnectedLayer) l7.getNextLayer();
		FullyConnectedLayer l9 = (FullyConnectedLayer) l8.getNextLayer();

		l2.setActivationType(ActivationType.RELU);
		l4.setActivationType(ActivationType.RELU);
		l6.setActivationType(ActivationType.RELU);
		l8.setActivationType(ActivationType.RELU);
		l9.setActivationType(ActivationType.NONE);
		l3.setPoolingType(PoolingType.MAX);
		l5.setPoolingType(PoolingType.AVER);
		l7.setPoolingType(PoolingType.AVER);
		cnn.setInputShape(new int[] {32, 32});

		l2.initializeWeights(0.0001f, 0.01f);
		l4.initializeWeights(0.01f, 0.01f);
		l6.initializeWeights(0.01f, 0.01f);
		l8.initializeWeights(0.1f, 0.01f);
		l9.initializeWeights(0.1f, 0.01f);


		
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		String path = Paths.get(System.getProperty("user.dir"), "..", "..", "..","datasets", "CIFAR", "cifar-10-batches-bin").toString();
		CIFAR10DataProvider tp = new CIFAR10DataProvider(path, batchSize, DatasetType.TRAINING_ALL, false);


		
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		
		cnn.forwardPass(tin);		
		cnn.backPropagation(tout, costType);
		int i = 0;
		System.out.printf(
				"l2 weights %f gradient %f \n"
				+ "l4 weights %f gradient %f \n"
				+ "l6 weights %f gradient %f \n"
				+ "l8 weights %f gradient %f \n"
				+ "l9 weights %f gradient %f \n", 
				l2.getWeight()[i], l2.getGradients()[i],
				l4.getWeight()[i], l4.getGradients()[i],
				l6.getWeight()[i], l6.getGradients()[i],
				l8.getWeight()[i], l8.getGradients()[i],
				l9.getWeight()[i], l9.getGradients()[i]
				);
		
		Layer l = l2;
		float g1 = l.getGradients()[i];
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.021f;
		weights[i] = (float) (w - e);
		l.setWeight(weights);
		cnn.forwardPass(tin);
		cnn.calCostErr(tout, costType);
		float c1 = cnn.getCost();
		weights[i] = (float) (w + e);
		l.setWeight(weights);
		cnn.forwardPass(tin);
		cnn.calCostErr(tout, costType);
		float c2 = cnn.getCost();
		double g2 = (c2 - c1)/(2 * e);
		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g2/g1, 0.0015);
		cnn.releaseCLMem();
//		OpenCL.releaseAll();
	}	
	
	@Test
	public void gradientCheckPadding() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
		int[][] para = {{3, 0, 0, 0}, {3, 3, 3, 1}, {5, 3, 3, 1}, {4, 3, 3, 1}, {10}};
		boolean addBias = true;
		boolean useOpenCL = true;
		int costType = 0;
		int batchSize = 10;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, true, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		ConvolutionalLayer l3 = (ConvolutionalLayer) l2.getNextLayer();
		ConvolutionalLayer l4 = (ConvolutionalLayer) l3.getNextLayer();
		l2.initializeWeights(0.0125f, 0);
		l3.initializeWeights(0.125f, 0);
		l4.initializeWeights(0.25f, 0);
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		cnn.setInputShape(new int[] {32, 32});
		String path = Paths.get(System.getProperty("user.dir"), "..", "..", "..","datasets", "CIFAR", "cifar-10-batches-bin").toString();
		CIFAR10DataProvider tp = new CIFAR10DataProvider(path, batchSize, DatasetType.TRAINING_ALL, false);


		
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		cnn.forwardPass(tin);
		
		cnn.backPropagation(tout, costType);
		int i = 10;
		Layer l = l2;
		float g1 = l.getGradients()[i];
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.000625f;
		weights[i] = (float) (w - e);
		l.setWeight(weights);
		cnn.forwardPass(tin);
		cnn.calCostErr(tout, costType);
		float c1 = cnn.getCost();
//		float[] a1 = l.getActivations();

		weights[i] = (float) (w + e);
		l.setWeight(weights);

		cnn.forwardPass(tin);
		cnn.calCostErr(tout, costType);
		float c2 = cnn.getCost();
		double g2 = (c2 - c1)/(2 * e);
//		float[] a2 = l.getActivations();
//		assertArrayEquals("!!",a1,a2, 0.0001f);

		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 0.0015);
//		assertEquals(0, (g1-g2)/(Math.abs(g1)+Math.abs(g2)), 0.000015);
		cnn.releaseCLMem();
//		OpenCL.releaseAll();
	}
	@Test
	public void testOpenCL() {
		int[][] para = {{3, 0, 0, 0}, {2, 3, 3, 1}, {20, 3, 3, 1}, {10}};
//		int[][] para = {{2, 0, 0, 0}, {2, 2, 2, 1}, {2, 2, 2, 1}, {10}};
		boolean addBias = true;
		boolean useOpenCL = false;
		int batchSize = 10;
		int costType = 1;
		String path = Paths.get(System.getProperty("user.dir"), "..", "..", "..","datasets", "CIFAR", "cifar-10-batches-bin").toString();
		CIFAR10DataProvider tp = new CIFAR10DataProvider(path, batchSize, DatasetType.TRAINING_ALL, false);
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
//		float[] tin = {	0.1f, 0.2f, 0.3f,
//				0.4f, 0.5f, 0.6f,
//				0.7f, 0.8f, 0.9f,
//				1.1f, 1.2f, 1.3f,
//				1.4f, 1.5f, 1.6f,
//				1.7f, 1.8f, 1.9f,
//				2.1f, 2.2f, 2.3f,
//				2.4f, 2.5f, 2.6f,
//				2.7f, 2.8f, 2.9f,
//				3.1f, 3.2f, 3.3f,
//				3.4f, 3.5f, 3.6f,
//				3.7f, 3.8f, 3.9f,
//			};
//		float[] tout = new float[20];
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		Layer l1 = cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		cnn.setInputShape(new int[] {32, 32});
//		cnn.setInputShape(new int[] {3, 3});
		cnn.forwardPass(tin);
		float[] a1 = l2.getActivations();
		
		useOpenCL = true;
		ConvolutionalNeuralNetwork cnn1 = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		Layer l11 = cnn1.getInputLayer();
		Layer l12 = l11.getNextLayer();
		Layer l13 = l12.getNextLayer();
		Layer l14 = l13.getNextLayer();
		cnn1.setInputShape(new int[] {32, 32});
//		cnn1.setInputShape(new int[] {3, 3});
		cnn1.forwardPass(tin);
		float[] a2 = l12.getActivations();
//		System.out.println("a1 "+ a1.length + Arrays.toString(a1));
//		System.out.println("a2 "+ a2.length + Arrays.toString(a2));

		assertArrayEquals("!!",a1,a2, 0.0001f);
	
		cnn.backPropagation(tout, costType);
		cnn1.backPropagation(tout, costType);
//		float[] e1 = l3.getPrevErrors();
//		float[] e2 = l13.getPrevErrors();
//		System.out.println("e1 "+ e1.length + Arrays.toString(e1));
//		System.out.println("e2 "+ e2.length + Arrays.toString(e2));
//		assertArrayEquals("!!",e1,e2, 0.0001f);
		float[] g1 = l3.getGradients();
		float[] g2 = l13.getGradients();
//		System.out.println("g1 "+ g1.length + Arrays.toString(g1));
//		System.out.println("g2 "+ g2.length + Arrays.toString(g2));
		assertArrayEquals("!!",g1,g2, 0.001f);
//		OpenCL.releaseAll();

	}

}
