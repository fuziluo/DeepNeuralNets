package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.ConvolutionalLayer;
import com.changjinxiong.deepneuralnets.nn.ConvolutionalNeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.FeatureMapLayer;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
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
							};
		cl2.setWeight(weights);
		cl2.forwardPass();
		float[] act = cl2.getActivations();
		float[] actCorrect = {	2.74f, 2.94f, 
								3.34f, 3.54f, 
								6.94f, 7.54f, 
								8.74f, 9.34f, 
								6.74f, 6.94f, 
								7.34f, 7.54f, 
								18.94f, 19.54f, 
								20.74f, 21.34f};
		for (int i = 0; i < actCorrect.length; i++) {
			actCorrect[i] = (float) (1 / (1 + Math.exp(-actCorrect[i])));
		}
		System.out.println(Arrays.toString(actCorrect));
		System.out.println(Arrays.toString(act));
		assertArrayEquals("!!",actCorrect,act, 0.0001f);

	}

	@Test
	public void testConvLayerForward1() {
		boolean useOpenCL = true;
		ConvolutionalLayer cl1 = new ConvolutionalLayer(2, 0, 0, 0, null, null, false, useOpenCL);
		ConvolutionalLayer cl2 = new ConvolutionalLayer(2, 2, 2, 2, cl1, null, true, useOpenCL);
		cl1.setNextLayer(cl2);
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

	}
	
	@Test
	public void testConstructionCNN() {
		int[][] para = {{2, 0, 0, 0}, {4, 3, 3, 1}, {3, 2, 2, 2}, {4}, {5}};
		boolean addBias = true;
		boolean useOpenCL = false;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		Layer l5 = l4.getNextLayer();
		l1.setInputShape(new int[] {4, 4});
		
		assertEquals(l2.getWeight().length, 40, 0);
		assertEquals(l3.getWeight().length, 15, 0);
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
		cnn.fordwardPass(testInput);
		assertEquals(l4.getWeight().length, 16, 0);
		float[] act = l5.getActivations();
//		System.out.println(Arrays.toString(act));
	}

	@Test
	public void testForwardPassCNN() {
		int[][] para = {{2, 0, 0, 0}, {2, 3, 3, 1}, {4}};
		boolean addBias = true;
		boolean useOpenCL = true;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		cnn.setInputShape(new int[] {4, 4});
		assertEquals(l2.getWeight().length, 20, 0);
//		assertNull(l3.getWeight());
		
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
		
		cnn.fordwardPass(testInput);
		assertEquals(l3.getWeight().length, 36, 0);
		float[] act2 = l2.getActivations();
		float[] act3 = l3.getActivations();
//		System.out.println(Arrays.toString(act2));
		System.out.println(Arrays.toString(act3));

		assertArrayEquals("!!",act3,actCorrect, 0.0001f);
	}

	@Test
	public void gradientCheck() {
//		int[][] para = {{1, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
		int[][] para = {{3, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
		boolean addBias = true;
		boolean useOpenCL = true;
		int batchSize = 100;
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		FeatureMapLayer l1 = (FeatureMapLayer) cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
//		cnn.setInputShape(new int[] {28, 28});
//		MnistDataProvider tp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		cnn.setInputShape(new int[] {32, 32});
		CIFAR10DataProvider tp = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);

		
		float[] tin = tp.getNextbatchInput();

		float[] tout = tp.getNextBatchLabel();
		cnn.fordwardPass(tin);
//		float c1 = mlp.getCost(tout);
		cnn.backPropagation(tout, 0);
		int i = 0;
		Layer l = l3;
		float g1 = l.getGradients()[i];
		double w = l.getWeight()[i];
		double e = 0.005f;
		l.getWeight()[i] = (float) (w - e);
//		System.out.println(l.getWeight()[i]);
		cnn.fordwardPass(tin);
		float c1 = cnn.getCost(tout, 0);
		float[] a1 = l.getActivations();

//		System.out.println(Arrays.toString(l.getWeight()));
//		System.out.println(l.getWeight()[i]);
		l.getWeight()[i] = (float) (w + e);
//		System.out.println(Arrays.toString(l.getWeight()));
//		System.out.println(l.getWeight()[i]);
		cnn.fordwardPass(tin);
		float c2 = cnn.getCost(tout, 0);
		double g2 = (c2 - c1)/(2 * e);
		float[] a2 = l.getActivations();
//		assertArrayEquals("!!",a1,a2, 0.0001f);

		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 0.0015);

	}

	@Test
	public void testOpenCL() {
		int[][] para = {{3, 0, 0, 0}, {2, 3, 3, 1}, {2, 3, 3, 1}, {10}};
//		int[][] para = {{2, 0, 0, 0}, {2, 2, 2, 1}, {2, 2, 2, 1}, {10}};
		boolean addBias = true;
		boolean useOpenCL = false;
		int batchSize = 100;
		CIFAR10DataProvider tp = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);
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
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		Layer l1 = cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		Layer l3 = l2.getNextLayer();
		cnn.setInputShape(new int[] {32, 32});
//		cnn.setInputShape(new int[] {3, 3});
		cnn.fordwardPass(tin);
		float[] a1 = l2.getActivations();
		
		useOpenCL = true;
		ConvolutionalNeuralNetwork cnn1 = new ConvolutionalNeuralNetwork(para, addBias, useOpenCL);
		Layer l11 = cnn1.getInputLayer();
		Layer l12 = l11.getNextLayer();
		Layer l13 = l12.getNextLayer();
		cnn1.setInputShape(new int[] {32, 32});
//		cnn1.setInputShape(new int[] {3, 3});
		cnn1.fordwardPass(tin);
		float[] a2 = l12.getActivations();
//		System.out.println("a1 "+ a1.length + Arrays.toString(a1));
//		System.out.println("a2 "+ a2.length + Arrays.toString(a2));

		assertArrayEquals("!!",a1,a2, 0.0001f);
	
		cnn.backPropagation(tout, 0);
		cnn1.backPropagation(tout, 0);
		float[] e1 = l3.getPrevErrors();
		float[] e2 = l13.getPrevErrors();
//		System.out.println("e1 "+ e1.length + Arrays.toString(e1));
//		System.out.println("e2 "+ e2.length + Arrays.toString(e2));
		assertArrayEquals("!!",e1,e2, 0.0001f);
		float[] g1 = l2.getGradients();
		float[] g2 = l12.getGradients();
//		System.out.println("g1 "+ g1.length + Arrays.toString(g1));
//		System.out.println("g2 "+ g2.length + Arrays.toString(g2));
		assertArrayEquals("!!",g1,g2, 0.0001f);

	}

}
