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
/**
 * 
 * @author jxchang
 *
 */
public class TestMLP {

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
	}

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testWeightsSize() {
		//check weights
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{10,20,30,20,10,1}, true);
		assertNull(mlp.getInputLayer().getWeight()) ;
		assertEquals(mlp.getInputLayer().getNextLayer().getWeight().length, 11*20, 0);
		assertEquals(mlp.getInputLayer().getNextLayer().getNextLayer().getWeight().length, 21*30, 0);
		assertEquals(mlp.getOutputLayer().getPreviousLayer().getPreviousLayer().getWeight().length, 31*20, 0);
		assertEquals(mlp.getOutputLayer().getPreviousLayer().getWeight().length, 21*10, 0);
		assertEquals(mlp.getOutputLayer().getWeight().length, 11*1, 0);		
	}
	@Test
	public void testForwardPass() {
		//check forward pass without bias
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{2,2,1}, false);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();
		l3.getWeight()[0] = 0.1f;
		l3.getWeight()[1] = 0.2f;
		l2.getWeight()[0] = 0.1f;
		l2.getWeight()[1] = 0.2f;
		l2.getWeight()[2] = 0.3f;
		l2.getWeight()[3] = 0.4f;		
		mlp.fordwardPass(new float[] {1, 2}, false);
//		System.out.println(Arrays.toString(l2.getWeight()));
		assertEquals(l2.getActivations()[0], 0.622459, 0.00001);
		assertEquals(l2.getActivations()[1], 0.750260, 0.00001);
		assertEquals(l3.getActivations()[0], 0.552876, 0.00001);
		//test OpenCL
		mlp.fordwardPass(new float[] {1, 2}, true);
		assertEquals(l2.getActivations()[0], 0.622459, 0.00001);
		assertEquals(l2.getActivations()[1], 0.750260, 0.00001);
		assertEquals(l3.getActivations()[0], 0.552876, 0.00001);
		
		//check forward pass with bias
		mlp = new MultiLayerPerceptron(new int[]{2,2,1}, true);
		l3 = mlp.getOutputLayer();
		l2 = mlp.getOutputLayer().getPreviousLayer();
		l1 = mlp.getInputLayer();
		l3.getWeight()[0] = 0.1f;
		l3.getWeight()[1] = 0.2f;
		l3.getWeight()[2] = 0.3f;
		
		l2.getWeight()[0] = 0.1f;
		l2.getWeight()[1] = 0.2f;
		l2.getWeight()[2] = 0.3f;
		l2.getWeight()[3] = 0.4f;		
		l2.getWeight()[4] = 0.5f;		
		l2.getWeight()[5] = 0.6f;		
		mlp.fordwardPass(new float[] {1, 2, 1}, false);
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
		
		//check forward pass with bias using OpenCL
		mlp = new MultiLayerPerceptron(new int[]{2,2,1}, true);
		l3 = mlp.getOutputLayer();
		l2 = mlp.getOutputLayer().getPreviousLayer();
		l1 = mlp.getInputLayer();
		l3.getWeight()[0] = 0.1f;
		l3.getWeight()[1] = 0.2f;
		l3.getWeight()[2] = 0.3f;
		
		l2.getWeight()[0] = 0.1f;
		l2.getWeight()[1] = 0.2f;
		l2.getWeight()[2] = 0.3f;
		l2.getWeight()[3] = 0.4f;		
		l2.getWeight()[4] = 0.5f;		
		l2.getWeight()[5] = 0.6f;		
		mlp.fordwardPass(new float[] {1, 2, 1}, true);
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
		//check cost
		float cost = mlp.getCost(new float[]  {1});
		assertEquals(cost, 0.4573, 0.0001);
	}

		
	@Test
	public void testBackpropagation() {
		
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{2,2,1}, true);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();

		l3.getWeight()[0] = 0.1f;
		l3.getWeight()[1] = 0.2f;
		l3.getWeight()[2] = 0.3f;
		
		l2.getWeight()[0] = 0.1f;
		l2.getWeight()[1] = 0.2f;
		l2.getWeight()[2] = 0.3f;
		l2.getWeight()[3] = 0.4f;		
		l2.getWeight()[4] = 0.5f;		
		l2.getWeight()[5] = 0.6f;		
		mlp.fordwardPass(new float[] {1, 2, 1}, true);
		//check back propagation 
		//gradient checking verify correctness of weight derivatives

		mlp.backPropagation(new float[] {1}, false);
		assertArrayEquals("!!",l2.getGradients(),new float[] {-0.007850246f, -0.015700491f, -0.007850246f, -0.007706293f, -0.015412586f, -0.007706293f}, 0.0001f);
		assertArrayEquals("!!",l3.getGradients(),new float[] {-0.2532129f, -0.32324263f, -0.36698878f}, 0.0001f);

		//check back propagation with OpenCL
		//gradient checking verify correctness of weight derivatives
		mlp.backPropagation(new float[] {1}, true);
		assertArrayEquals("!!",l2.getGradients(),new float[] {-0.007850246f, -0.015700491f, -0.007850246f, -0.007706293f, -0.015412586f, -0.007706293f}, 0.0001f);
		assertArrayEquals("!!",l3.getGradients(),new float[] {-0.2532129f, -0.32324263f, -0.36698878f}, 0.0001f);
		
	}
	@Test
	public void testBackpropagationBatch() {
		
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{2,42,2}, true);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();
		
		//check forward pass with bias in batch
//		l3.getWeight()[0] = 0.1f;
//		l3.getWeight()[1] = 0.2f;
//		l3.getWeight()[2] = 0.3f;
//		
//		l2.getWeight()[0] = 0.1f;
//		l2.getWeight()[1] = 0.2f;
//		l2.getWeight()[2] = 0.3f;
//		l2.getWeight()[3] = 0.4f;		
//		l2.getWeight()[4] = 0.5f;		
//		l2.getWeight()[5] = 0.6f;		
//		mlp.fordwardPass(new float[] {1, 2, 1, 3, 4, 1}, false);
//		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
//		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
//		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
//		mlp.fordwardPass(new float[] {1, 2, 1, 3, 4, 1}, true);
//		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
//		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
//		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);

		//check cost
//		float cost = mlp.getCost(new float[]  {1, 1});
//		assertEquals(cost, 0.451691, 0.0001);
		//check back propagation 
		//gradient checking verify correctness of weight derivatives
		float[] tin = new float[] {	
				1, 2, 1, 3, 4, 1, 1, 2, 1, 3, 4, 1, 
				5, 2, 1, 3, 5, 1, 1, 2, 1, 3, 6, 1, 
				1, 2, 1, 3, 4, 1, 1, 0, 1, 3, 4, 1, 
				1, 2, 1, 3, 0, 1, 1, 3, 1, 3, 9, 1,
				7, 9, 1, 7, 4, 1, 1, 2, 1, 3, 4, 1, 
				1, 2, 1, 3, 2, 1, 1, 2, 1, 3, 5, 1,
				1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 4, 1,
				1, 2, 1, 3, 4, 1, 1, 1, 1, 3, 8, 1,
				1, 2, 1 
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
		mlp.fordwardPass(tin , false);
		float c1 = mlp.getCost(tout);
		mlp.backPropagation(tout, false);
//		mlp.updateWeights(0.01f);
//		mlp.fordwardPass(new float[] {1, 2, 1, 3, 1, 1}, false);
//		mlp.backPropagation(new float[] {1, 0}, false);
//		mlp.updateWeights(0.01f);
		float[] g1 = l2.getGradients();
		float[] w1 = l2.getWeight();
		System.out.println(Arrays.toString(g1));
//		System.out.println(Arrays.toString(w1));
		
		MultiLayerPerceptron mlp1 = new MultiLayerPerceptron(new int[]{2,42,2}, true);
		l3 = mlp1.getOutputLayer();
		l2 = mlp1.getOutputLayer().getPreviousLayer();
		l1 = mlp1.getInputLayer();
			

		mlp1.fordwardPass(tin, true);
		float c2 = mlp1.getCost(tout);
		mlp1.backPropagation(tout, true);
//		mlp1.updateWeights(0.01f);
//		mlp1.fordwardPass(new float[] {1, 2, 1, 3, 1, 1}, true);
//		mlp1.backPropagation(new float[] {1, 0}, true);
//		mlp1.updateWeights(0.01f);

		float[] g2 = l2.getGradients();
		float[] w2 = l2.getWeight();
		System.out.println(Arrays.toString(g2));
		System.out.println(l2.getBatchSize());
//		System.out.println(Arrays.toString(w2));

		assertArrayEquals("!!",g1,g2, 0.00001f);
		assertArrayEquals("!!",w1,w2, 0.00001f);
		assertEquals(c1, c2, 0.0001);

	}

}
