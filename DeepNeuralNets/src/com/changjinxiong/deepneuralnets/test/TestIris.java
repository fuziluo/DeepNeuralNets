package com.changjinxiong.deepneuralnets.test;
import static java.lang.Math.*;
import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import com.changjinxiong.deepneuralnets.test.IrisDataProvider;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;

public class TestIris {

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
	public void testIrisDataProvider() {
		IrisDataProvider tp = new IrisDataProvider(1, false);
		for (int i = 0; i < 300; i ++) {
			
			System.out.printf("%d %s %s",i+1,Arrays.toString(tp.getNextbatchInput()),Arrays.toString(tp.getNextBatchLabel()));
			System.out.println();

		}

	}
	@Test
	public void testOpenCL() {
		boolean useOpenCL = false;
		int batchSize = 33;
		//without OpenCL
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{4,32,3}, true, useOpenCL); 
		IrisDataProvider tp = new IrisDataProvider(batchSize, false);
		mlp.train(tp, 0, 0.005f, 0, 0, 0, 0, 1);
		float[] t = tp.getNextBatchLabel();
		float c1 = mlp.getCost();

		//with OpenCL
		useOpenCL = true;
		mlp = new MultiLayerPerceptron(new int[]{4,32,3}, true, useOpenCL); 
		IrisDataProvider tp1 = new IrisDataProvider(batchSize, false);
		mlp.train(tp1, 0, 0.005f, 0, 0, 0, 0, 1);
		float[] t1 = tp1.getNextBatchLabel();
		float c2 = mlp.getCost();
		
		assertEquals(c1, c2, 0.0001);

		
		
		
	}
	@Test
	public void test() {
		boolean useOpenCL = true;
		int costType = 0; //cross entropy
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{4,33,3}, true, useOpenCL); //overfitting
		int batchSize = 150;
		IrisDataProvider tp = new IrisDataProvider(batchSize, false);
		mlp.train(tp, costType, 0.5f, 0.0f, 0, 0, 0, 10000);
		//test
		IrisDataProvider tp1 = new IrisDataProvider(150, false);
		float errorRate = mlp.test(tp1);
		assertEquals(0, errorRate, 0.01);

	}

}
