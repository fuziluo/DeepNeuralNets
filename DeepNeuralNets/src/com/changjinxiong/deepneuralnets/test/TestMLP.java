package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.FullyConnectedLayer;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
/**
 * 
 * @author jxchang
 *
 */
public class TestMLP {

	
	@Test
	public void testWeightsSize() {
		//check weights
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{10,20,30,20,10,1}, true, false);
		assertEquals(mlp.getInputLayer().getNextLayer().getWeight().length, 11*20, 0);
		assertEquals(mlp.getInputLayer().getNextLayer().getNextLayer().getWeight().length, 21*30, 0);
		assertEquals(mlp.getOutputLayer().getPreviousLayer().getPreviousLayer().getWeight().length, 31*20, 0);
		assertEquals(mlp.getOutputLayer().getPreviousLayer().getWeight().length, 21*10, 0);
		assertEquals(mlp.getOutputLayer().getWeight().length, 11*1, 0);		
	}
	@Test
	public void testForwardPass() {
		boolean useOpenCL = true;
		//check forward pass without bias
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{2,2,1}, false, useOpenCL);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();
		l3.setWeight(new float[] {0.1f, 0.2f});
		l2.setWeight(new float[] {0.1f, 0.2f, 0.3f, 0.4f});
//		mlp.fordwardPass(new float[] {1, 2});
////		System.out.println(Arrays.toString(l2.getWeight()));
//		assertEquals(l2.getActivations()[0], 0.622459, 0.00001);
//		assertEquals(l2.getActivations()[1], 0.750260, 0.00001);
//		assertEquals(l3.getActivations()[0], 0.552876, 0.00001);
		//test OpenCL
		mlp.fordwardPass(new float[] {1, 2});
		assertEquals(l2.getActivations()[0], 0.622459, 0.00001);
		assertEquals(l2.getActivations()[1], 0.750260, 0.00001);
		assertEquals(l3.getActivations()[0], 0.552876, 0.00001);
		
		//check forward pass with bias
		mlp = new MultiLayerPerceptron(new int[]{2,2,1}, true, useOpenCL);
		l3 = mlp.getOutputLayer();
		l2 = mlp.getOutputLayer().getPreviousLayer();
		l1 = mlp.getInputLayer();
		l3.setWeight(new float[] {0.1f, 0.2f, 0.3f});
		l2.setWeight(new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
		mlp.fordwardPass(new float[] {1, 2});
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
		
		//check cost
		float cost = mlp.getCost(new float[] {1}, 0);
		assertEquals(cost, 0.4573, 0.0001);
	}

		
	@Test
	public void testBackpropagation() {
		boolean useOpenCL = true;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{2,2,1}, true, useOpenCL);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();

		l3.setWeight(new float[] {0.1f, 0.2f, 0.3f});
		l2.setWeight(new float[] {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});

		mlp.fordwardPass(new float[] {1, 2});
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);

		//check back propagation with OpenCL
		//gradient checking verify correctness of weight derivatives
		mlp.backPropagation(new float[] {1}, 0);
		assertArrayEquals("!!",l2.getGradients(),new float[] {-0.007850246f, -0.015700491f, -0.007850246f, -0.007706293f, -0.015412586f, -0.007706293f}, 0.0001f);
		assertArrayEquals("!!",l3.getGradients(),new float[] {-0.2532129f, -0.32324263f, -0.36698878f}, 0.0001f);
		
	}
	@Test
	public void testBackpropagationBatch() {
		boolean addBias = true;
		boolean useOpenCL = false;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{2,42,2}, addBias, useOpenCL);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();
		
		//check back propagation 
		float[] tin = new float[] {	
				1, 2, 1, 3, 4, 1, 1, 2,  
				5, 2, 1, 3, 5, 1, 1, 2,  
				1, 2, 1, 3, 4, 1, 1, 0,  
				1, 2, 1, 3, 0, 1, 1, 3,
				7, 9, 1, 7, 4, 1, 1, 2,  
				1, 2, 1, 3, 2, 1, 1, 2, 
				1, 1, 1, 3, 3, 1, 1, 1, 
				1, 2, 1, 3, 4, 1, 1, 1, 
				1, 2
									};
		float[] tout = new float[] {
				1, 1, 1, 1, 1, 0, 1, 1, 
				1, 0, 1, 1, 1, 1, 1, 1, 
				1, 1, 0, 0, 1, 0, 1, 1, 
				1, 0, 1, 1, 1, 1, 1, 1,
				1, 1, 1, 1, 1, 0, 1, 1, 
				1, 0, 1, 0, 1, 1, 1, 1, 
				1, 1, 0, 1, 1, 0, 1, 1, 
				1, 1, 1, 1, 1, 1, 1, 1,
				1, 1
									
									};
		mlp.fordwardPass(tin);
		float c1 = mlp.getCost(tout, 0);
		float[] g1 = l2.getGradients();
		float[] g11 = l3.getGradients();
		float[] a1 = l2.getActivations();
		float[] a11 = l3.getActivations();
		
		System.out.println("a1 "+Arrays.toString(a1));
		mlp.backPropagation(tout, 0);
//		mlp.updateWeights(0.01f);
//		mlp.fordwardPass(new float[] {1, 2, 1, 3, 1, 1}, false);
//		mlp.backPropagation(new float[] {1, 0}, false);
//		mlp.updateWeights(0.01f);
//		float[] w1 = l2.getWeight();
		System.out.println("g11 "+Arrays.toString(g11));
//		System.out.println(Arrays.toString(w1));
		
		
		useOpenCL = true;
		NeuralNetwork mlp1 = new MultiLayerPerceptron(new int[]{2,42,2}, addBias, useOpenCL);
		l3 = mlp1.getOutputLayer();
		l2 = mlp1.getOutputLayer().getPreviousLayer();
		l1 = mlp1.getInputLayer();
			

		mlp1.fordwardPass(tin);
		float c2 = mlp1.getCost(tout, 0);
		float[] g2 = l2.getGradients();
		float[] g12 = l3.getGradients();
		float[] a2 = l2.getActivations();
		float[] a12 = l3.getActivations();
		System.out.println("a2 "+Arrays.toString(a2));
		System.out.println("g12 "+Arrays.toString(g12));
		mlp1.backPropagation(tout, 0);
//		mlp1.updateWeights(0.01f);
//		mlp1.fordwardPass(new float[] {1, 2, 1, 3, 1, 1}, true);
//		mlp1.backPropagation(new float[] {1, 0}, true);
//		mlp1.updateWeights(0.01f);

//		float[] w2 = l2.getWeight();
		g12 = l3.getGradients();
		System.out.println("g12 "+Arrays.toString(g12));
		System.out.println(l2.getBatchSize());
//		System.out.println(Arrays.toString(w2));

		assertArrayEquals("!!",a1,a2, 0.00001f);
		assertArrayEquals("!!",a11,a12, 0.00001f);
		assertArrayEquals("!!",g11,g12, 0.00001f);
		assertArrayEquals("!!",g1,g2, 0.00001f);
//		assertArrayEquals("!!",w1,w2, 0.00001f);
		assertEquals(c1, c2, 0.0001);

	}
	
	@Test
	public void gradientCheck() {
		boolean useOpenCL = true;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{784,300,10}, true, useOpenCL);
		Layer l3 = mlp.getOutputLayer();
		int costType = 1;
		MnistDataProvider mnistTraining = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", 100, false);
//		float[] tin = {	0.1f, 0.2f, 0.3f, 1,
//						0.4f, 0.5f, 0.6f, 1,
//						0.7f, 0.8f, 0.9f, 1};
//		float[] tout = {1, 0,
//						0, 1,
//						1, 1};
		float[] tin = mnistTraining.getNextbatchInput();
		float[] tout = mnistTraining.getNextBatchLabel();
		mlp.fordwardPass(tin);
//		float c1 = mlp.getCost(tout);
		mlp.backPropagation(tout, costType);
		int i = 1;
		float g1 = l3.getGradients()[i];
		double w = l3.getWeight()[i];
		double e = 0.01f;
		float[] tempW = l3.getWeight();
		tempW[i] = (float) (w - e);
		l3.setWeight(tempW);
//		System.out.println(l3.getWeight()[i]);
		mlp.fordwardPass(tin);
		float c1 = mlp.getCost(tout, costType);
		tempW[i] = (float) (w + e);
		l3.setWeight(tempW);
//		System.out.println(l3.getWeight()[i]);
		mlp.fordwardPass(tin);
		float c2 = mlp.getCost(tout, costType);
		double g2 = (c2 - c1)/(2 * e);
		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 5e-4);

	}
	
	@Test
	public void testWeightsSaveLoad() {
		boolean useOpenCL = true;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{2,2,1}, false, useOpenCL);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		l3.setWeight(new float[] {0.1f, 0.2f});
		l2.setWeight(new float[] {0.1f, 0.2f, 0.3f, 0.4f});
		
		assertArrayEquals("Initial check wrong l2", new float[] {0.1f, 0.2f, 0.3f, 0.4f}, l2.getWeight(), 0.00001f);
		assertArrayEquals("Initial check wrong l3", new float[] {0.1f, 0.2f}, l3.getWeight(), 0.00001f);

		
		String path = "/home/jxchang/project/DeepLearningInJava/DeepNeuralNets/DeepNeuralNets/test/.MultiLayerPerceptron.weights";
		mlp.saveWeights(path);
	
		l3.setWeight(new float[] {1.1f, 1.2f});
		l2.setWeight(new float[] {1.1f, 1.2f, 1.3f, 1.4f});
		
		assertArrayEquals("Second check wrong l2", new float[] {1.1f, 1.2f, 1.3f, 1.4f}, l2.getWeight(), 0.00001f);
		assertArrayEquals("Second check wrong l3", new float[] {1.1f, 1.2f}, l3.getWeight(), 0.00001f);
		
		mlp.loadWeights(path);
		
		assertArrayEquals("Final check wrong l2", new float[] {0.1f, 0.2f, 0.3f, 0.4f}, l2.getWeight(), 0.00001f);
		assertArrayEquals("Final check wrong l3", new float[] {0.1f, 0.2f}, l3.getWeight(), 0.00001f);

	}

}
