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
import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
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
		l2.setActivationType(ActivationType.SIGMOID);
		l3.setActivationType(ActivationType.SIGMOID);
//		mlp.fordwardPass(new float[] {1, 2});
////		System.out.println(Arrays.toString(l2.getWeight()));
//		assertEquals(l2.getActivations()[0], 0.622459, 0.00001);
//		assertEquals(l2.getActivations()[1], 0.750260, 0.00001);
//		assertEquals(l3.getActivations()[0], 0.552876, 0.00001);
		//test OpenCL
		mlp.forwardPass(new float[] {1, 2});
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
		l2.setActivationType(ActivationType.SIGMOID);
		l3.setActivationType(ActivationType.SIGMOID);
		mlp.forwardPass(new float[] {1, 2});
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
		
		//check cost
		mlp.calCostErr(new float[] {1}, 0);
		float cost = mlp.getCost();
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
		l2.setActivationType(ActivationType.SIGMOID);
		l3.setActivationType(ActivationType.SIGMOID);

		mlp.forwardPass(new float[] {1, 2});
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
		l2.setActivationType(ActivationType.SIGMOID);
		l3.setActivationType(ActivationType.SIGMOID);
		
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
		mlp.forwardPass(tin);
		float[] a1 = l2.getActivations();
		float[] a11 = l3.getActivations();
		mlp.backPropagation(tout, 0);
		mlp.calCostErr(tout, 0);
		float c1 = mlp.getCost();
		float[] g1 = l2.getGradients();
		float[] g11 = l3.getGradients();
//		float[] e1 = l3.getPrevErrors();
		
		
		useOpenCL = true;
		NeuralNetwork mlp1 = new MultiLayerPerceptron(new int[]{2,42,2}, addBias, useOpenCL);
		Layer l13 = mlp1.getOutputLayer();
		Layer l12 = mlp1.getOutputLayer().getPreviousLayer();
		Layer l11 = mlp1.getInputLayer();
		l12.setActivationType(ActivationType.SIGMOID);
		l13.setActivationType(ActivationType.SIGMOID);
			

		mlp1.forwardPass(tin);
		mlp1.calCostErr(tout, 0);
		float c2 = mlp1.getCost();
		float[] a2 = l12.getActivations();
		float[] a12 = l13.getActivations();
		mlp1.backPropagation(tout, 0);

		float[] g2 = l12.getGradients();
		float[] g12 = l13.getGradients();
//		float[] e2 = l13.getPrevErrors();
		System.out.println("g1 "+Arrays.toString(g1));
		System.out.println("g2 "+Arrays.toString(g2));
//		System.out.println("e1 "+Arrays.toString(e1));
//		System.out.println("e2 "+Arrays.toString(e2));

		assertArrayEquals("!!",a1,a2, 0.00001f);
		assertArrayEquals("!!",a11,a12, 0.00001f);
		assertArrayEquals("!!",g11,g12, 0.00001f);
//		assertArrayEquals("!!",e1,e2, 0.00001f);
		assertArrayEquals("!!",g1,g2, 0.00001f);
		assertEquals(c1, c2, 0.0001);

	}
	
	@Test
	public void gradientCheck() {
		boolean useOpenCL = true;
//		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{784,300,300,10}, true, useOpenCL);
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{8,10,5,3}, true, useOpenCL);
		Layer l4 = mlp.getOutputLayer();
		Layer l3 = l4.getPreviousLayer();
		Layer l2 = l3.getPreviousLayer();
		int costType = 0;
		int batchSize = 8;
		MnistDataProvider mnistTraining = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);

//		float[] tin = mnistTraining.getNextbatchInput();
//		float[] tout = mnistTraining.getNextBatchLabel();
		float[] tin = new float[] {	
				1, 2, 1, 3, 4, 1, 1, 2,  
				5, 2, 1, 3, 5, 1, 1, 2,  
				1, 2, 1, 3, 4, 1, 1, 0,  
				1, 2, 1, 3, 0, 1, 1, 3,
				7, 9, 1, 7, 4, 1, 1, 2,  
				1, 2, 1, 3, 2, 1, 1, 2, 
				1, 1, 1, 3, 3, 1, 1, 1, 
				1, 2, 1, 3, 4, 1, 1, 1, 
									};
		float[] tout = new float[] {
				1, 1, 1, 1, 1, 0, 1, 1, 
				1, 0, 1, 1, 1, 1, 1, 1, 
				1, 1, 0, 0, 1, 0, 1, 1, 									
									};
		
		mlp.forwardPass(tin);
		mlp.backPropagation(tout, costType);
		int i = 1;
		Layer l = l2;
		float g1 = l.getGradients()[i];
		double w = l.getWeight()[i];
		double e = 0.005f;
		float[] tempW = l.getWeight();
		tempW[i] = (float) (w - e);
		l.setWeight(tempW);
//		System.out.println(l3.getWeight()[i]);
		mlp.forwardPass(tin);
		mlp.calCostErr(tout, costType);
		float c1 = mlp.getCost();
		tempW[i] = (float) (w + e);
		l.setWeight(tempW);
//		System.out.println(l3.getWeight()[i]);
		mlp.forwardPass(tin);
		mlp.calCostErr(tout, costType);
		float c2 = mlp.getCost();
		double g2 = (c2 - c1)/(2 * e);
		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(1, g1/g2, 0.002);

	}

	@Test
	public void gradientCheckReLU() {
		boolean useOpenCL = true;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{784,300,10}, true, useOpenCL);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = l3.getPreviousLayer();
		l2.setActivationType(ActivationType.RELU);
		l3.setActivationType(ActivationType.NONE);
		int costType = 0;
		MnistDataProvider mnistTraining = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", 20, false);
//		float[] tin = {	0.1f, 0.2f, 0.3f, 1,
//						0.4f, 0.5f, 0.6f, 1,
//						0.7f, 0.8f, 0.9f, 1};
//		float[] tout = {1, 0,
//						0, 1,
//						1, 1};
		float[] tin = mnistTraining.getNextbatchInput();
		float[] tout = mnistTraining.getNextBatchLabel();
		mlp.forwardPass(tin);
//		float c1 = mlp.getCost(tout);
		mlp.backPropagation(tout, costType);
		int i = 2;
		Layer l = l3;
		float g1 = l.getGradients()[i];
		double w = l.getWeight()[i];
		double e = 0.01f;
		float[] tempW = l.getWeight();
		tempW[i] = (float) (w - e);
		l.setWeight(tempW);
//		System.out.println(l3.getWeight()[i]);
		mlp.forwardPass(tin);
		mlp.calCostErr(tout, costType);
		float c1 = mlp.getCost();
		tempW[i] = (float) (w + e);
		l.setWeight(tempW);
//		System.out.println(l3.getWeight()[i]);
		mlp.forwardPass(tin);
		mlp.calCostErr(tout, costType);
		float c2 = mlp.getCost();
		double g2 = (c2 - c1)/(2 * e);
		System.out.println(c1+" "+c2);
		System.out.println(g1+" "+g2);
		assertEquals(g2, g1, 5e-4);

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
