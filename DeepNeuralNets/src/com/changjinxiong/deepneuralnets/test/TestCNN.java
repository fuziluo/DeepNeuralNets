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
import com.changjinxiong.deepneuralnets.nn.WeightLayer;
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
							0.3f, 0.4f, 
							0.6f, 0.7f,
							0.8f, 0.9f, 0.5f,
							
							1.1f, 1.2f, 
							1.3f, 1.4f, 
							1.6f, 1.7f,
							1.8f, 1.9f, 1,
							};
		cl2.setWeight(weights);
		cl2.setActivationType(ActivationType.SIGMOID);
		cl2.forwardPass();
		float[] act = cl2.getActivations();
		float[] actCorrect = {	
				4.8400f,    5.2400f,
			    6.0400f,    6.4400f,
			    11.7400f,   12.9400f,
			    15.3400f,   16.5400f,
			    
			    12.8400f,   13.2400f,
			    14.0400f,   14.4400f,
			    35.7400f,   36.9400f,
			    39.3400f,   40.5400f,

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
							0.3f, 0.4f, 
							0.1f, 0.2f, 
							0.3f, 0.4f, 0.5f,
							0.6f, 0.7f,
							0.8f, 0.9f,
							0.6f, 0.7f,
							0.8f, 0.9f, 1,
							};
		cl2.setWeight(weights);
		cl2.setActivationType(ActivationType.SIGMOID);
		cl2.forwardPass();
		float[] act = cl2.getActivations();
		float[] actCorrect = {	    2.2400f,    2.4400f,    1.3200f,
								    2.8400f,    3.0400f,    1.5600f,
								    1.2600f,    1.3200f,    0.7800f,
								    5.9400f,    6.5400f,    3.7200f,
								    7.7400f,    8.3400f,    4.5600f,
								    4.2600f,    4.5200f,    2.6800f,
													    
								    2.2400f,    2.4400f,    1.3200f,
								    2.8400f,    3.0400f,    1.5600f,
								    1.2600f,    1.3200f,    0.7800f, 
								    5.9400f,    6.5400f,    3.7200f,
								    7.7400f,    8.3400f,    4.5600f,
								    4.2600f,    4.5200f,    2.6800f, 
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
		WeightLayer l2 = (WeightLayer) l1.getNextLayer();
		WeightLayer l3 = (WeightLayer) l2.getNextLayer();
		WeightLayer l4 = (WeightLayer) l3.getNextLayer();
		WeightLayer l5 = (WeightLayer) l4.getNextLayer();
		l1.setInputShape(new int[] {4, 4});

		assertEquals(l2.getWeight().length, 76, 0);
		assertEquals(l3.getWeight().length, 51, 0);
		assertEquals(l4.getWeight().length, 16, 0);
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
//		assertEquals(l4.getWeight().length, 16, 0);
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
		WeightLayer l2 = (WeightLayer) l1.getNextLayer();
		WeightLayer l3 = (WeightLayer) l2.getNextLayer();
		cnn.setInputShape(new int[] {4, 4});
		assertEquals(l2.getWeight().length, 38, 0);
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
		
		float[] w2 = {	0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, 
						0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, 1.0f,
						1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f, 1.7f, -1.8f, 1.9f, 
						1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f, 1.7f, -1.8f, 1.9f, 2.0f,
		};
		l2.setWeight(w2);
		float[] w3 = {	0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f,
						1.1f, -1.2f, 1.3f, -1.4f, 1.5f, -1.6f, 1.7f, -1.8f, 1.9f,
						2.1f, -2.2f, 2.3f, -2.4f, 2.5f, -2.6f, 2.7f, -2.8f, 2.9f,
						3.1f, -3.2f, 3.3f, -3.4f, 3.5f, -3.6f, 3.7f, -3.8f, 3.9f,
		};
		l3.setWeight(w3);
		
		float[] actCorrect = {	0.61802465f, 0.78727967f, 0.8943569f, 0.9508964f, 
								0.62346935f, 0.8107077f, 0.91720235f, 0.9662761f
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
		int batchSize = 10;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		ConvolutionalLayer l3 = (ConvolutionalLayer) l2.getNextLayer();
		ConvolutionalLayer l4 = (ConvolutionalLayer) l3.getNextLayer();
		l2.initWeightsGaussian(0.25f, 0, 0);
		l3.initWeightsGaussian(0.25f, 0, 0);
		l4.initWeightsGaussian(0.25f, 0, 0);
		
		l2.setActivationType(ActivationType.TANH);
		l3.setActivationType(ActivationType.TANH);
		l4.setActivationType(ActivationType.TANH);

		
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		cnn.setInputShape(new int[] {32, 32});
		String path = Paths.get(System.getProperty("user.dir"), "..", "..", "..","datasets", "CIFAR", "cifar-10-batches-bin").toString();
		CIFAR10DataProvider tp = new CIFAR10DataProvider(path, batchSize, DatasetType.TRAINING_ALL, false);


		
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		cnn.forwardPass(tin);
		
		cnn.backPropagation(tout, costType);
		int i = 27; //the index of bias
		WeightLayer l = l2;
		float g1 = l.getGradients()[i];
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.003f;
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
	public void gradientCheckPadding() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
		int[][] para = {{3, 0, 0, 0}, {3, 3, 3, 1}, {4, 3, 3, 1}, {4, 3, 3, 1}, {10}};
		boolean addBias = true;
		boolean useOpenCL = true;
		int costType = 0;
		int batchSize = 10;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, true, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		ConvolutionalLayer l2 = (ConvolutionalLayer) l1.getNextLayer();
		ConvolutionalLayer l3 = (ConvolutionalLayer) l2.getNextLayer();
		ConvolutionalLayer l4 = (ConvolutionalLayer) l3.getNextLayer();
		l2.initWeightsGaussian(0.25f, 0, 0);
		l3.initWeightsGaussian(0.25f, 0, 0);
		l4.initWeightsGaussian(0.25f, 0, 0);
		
		
		l2.setActivationType(ActivationType.TANH);
		l3.setActivationType(ActivationType.TANH);
		l4.setActivationType(ActivationType.TANH);


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
		WeightLayer l = l2;
		float g1 = l.getGradients()[i];
		float[] weights = l.getWeight();
		double w = weights[i];
		double e = 0.0003f;
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
		boolean addBias = true;
		boolean useOpenCL = false;
		int batchSize = 10;
		int costType = 1;
		String path = Paths.get(System.getProperty("user.dir"), "..", "..", "..","datasets", "CIFAR", "cifar-10-batches-bin").toString();
		CIFAR10DataProvider tp = new CIFAR10DataProvider(path, batchSize, DatasetType.TRAINING_ALL, false);
		float[] tin = tp.getNextbatchInput();
		float[] tout = tp.getNextBatchLabel();
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		Layer l1 = cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		cnn.setInputShape(new int[] {32, 32});
		cnn.forwardPass(tin);
		float[] a1 = l2.getActivations();
		
		useOpenCL = true;
		ConvolutionalNeuralNetwork cnn1 = new ConvolutionalNeuralNetwork(para, addBias, false, useOpenCL);
		Layer l11 = cnn1.getInputLayer();
		Layer l12 = l11.getNextLayer();
		Layer l13 = l12.getNextLayer();
		Layer l14 = l13.getNextLayer();
		cnn1.setInputShape(new int[] {32, 32});
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
		System.out.println("g1 "+ g1.length + Arrays.toString(g1));
		System.out.println("g2 "+ g2.length + Arrays.toString(g2));
		assertArrayEquals("!!",g1,g2, 0.001f);
//		OpenCL.releaseAll();

	}

}
