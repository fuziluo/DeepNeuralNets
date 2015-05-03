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
	public void test() {
		//check weights
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{10,20,30,20,10,1}, true);
		assertNull(mlp.getInputLayer().getWeight()) ;
		assertEquals(mlp.getInputLayer().getNextLayer().getWeight().length, 11*20, 0);
		assertEquals(mlp.getInputLayer().getNextLayer().getNextLayer().getWeight().length, 21*30, 0);
		assertEquals(mlp.getOutputLayer().getPreviousLayer().getPreviousLayer().getWeight().length, 31*20, 0);
		assertEquals(mlp.getOutputLayer().getPreviousLayer().getWeight().length, 21*10, 0);
		assertEquals(mlp.getOutputLayer().getWeight().length, 11*1, 0);

		//check forward pass without bias
		mlp = new MultiLayerPerceptron(new int[]{2,2,1}, false);
		Layer l3 = mlp.getOutputLayer();
		Layer l2 = mlp.getOutputLayer().getPreviousLayer();
		Layer l1 = mlp.getInputLayer();
		l3.getWeight()[0] = 0.1f;
		l3.getWeight()[1] = 0.2f;
		l2.getWeight()[0] = 0.1f;
		l2.getWeight()[1] = 0.2f;
		l2.getWeight()[2] = 0.3f;
		l2.getWeight()[3] = 0.4f;		
		mlp.fordwardPass(new float[] {1, 2});
//		System.out.println(Arrays.toString(l2.getWeight()));
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
		mlp.fordwardPass(new float[] {1, 2, 1});
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
		//check cost
		float cost = mlp.getCost(new float[]  {1});
		assertEquals(cost, 0.4573, 0.0001);
		//check back propagation 
		//gradient checking verify correctness of weight derivatives
		mlp.backPropagation(new float[] {1});
		assertArrayEquals("!!",l2.getGradients(),new float[] {-0.007850246f, -0.015700491f, -0.007850246f, -0.007706293f, -0.015412586f, -0.007706293f}, 0.0001f);
		assertArrayEquals("!!",l3.getGradients(),new float[] {-0.2532129f, -0.32324263f, -0.36698878f}, 0.0001f);

		
		//check forward pass with bias in batch
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
		mlp.fordwardPass(new float[] {1, 2, 1, 3, 4, 1});
		assertEquals(l2.getActivations()[0], 0.689974, 0.00001);
		assertEquals(l2.getActivations()[1], 0.880797, 0.00001);
		assertEquals(l3.getActivations()[0], 0.6330112, 0.00001);
		//check cost
		cost = mlp.getCost(new float[]  {1, 1});
		assertEquals(cost, 0.451691, 0.0001);
		//check back propagation 
		//gradient checking verify correctness of weight derivatives
		mlp.backPropagation(new float[] {1, 1});
		System.out.println(Arrays.toString(l2.getGradients()));
		System.out.println(Arrays.toString(l3.getGradients()));

//		assertArrayEquals("!!",l2.getGradients(),new float[] {0.007850246f, 0.015700491f, 0.007850246f, 0.007706293f, 0.015412586f, 0.007706293f}, 0.0001f);
//		assertArrayEquals("!!",l3.getGradients(),new float[] {0.2532129f, 0.32324263f, 0.36698878f}, 0.0001f);
		
		
//		fail("Not yet implemented");
	}

}
