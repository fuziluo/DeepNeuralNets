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
			
			System.out.printf("%d %s %s",i+1,Arrays.toString(tp.getNextbatchInput(true)),Arrays.toString(tp.getNextBatchLabel()));
			System.out.println();

		}

	}
	@Test
	public void testOpenCL() {
		int batchSize = 33;
		//without OpenCL
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{4,32,3}, true); //overfitting
		IrisDataProvider tp = new IrisDataProvider(batchSize, false);
		mlp.train(tp, 0.005f, 0, 0, 0, 1, false);
		float[] t = tp.getNextBatchLabel();
		float c1 = mlp.getCost(t);

		//with OpenCL
		mlp = new MultiLayerPerceptron(new int[]{4,32,3}, true); //overfitting
		IrisDataProvider tp1 = new IrisDataProvider(batchSize, false);
		mlp.train(tp1, 0.005f, 0, 0, 0, 1, true);
		float[] t1 = tp1.getNextBatchLabel();
		float c2 = mlp.getCost(t1);
		
		assertEquals(c1, c2, 0.0001);

		
		
		
	}
	@Test
	public void test() {
		boolean openCL = true;
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{4,33,3}, true); //overfitting
		int batchSize = 150;
		IrisDataProvider tp = new IrisDataProvider(batchSize, false);
		mlp.train(tp, 0.005f, 0.0f, 0, 0, 10000, openCL);
		//test
		IrisDataProvider tp1 = new IrisDataProvider(150, false);
//		mlp.fordwardPass(tp1.getNextbatchInput(true), false);
//		float[] a = mlp.getOutputLayer().getActivations();
//		float[] t = tp1.getNextBatchLabel();
//
//		for (int i = 0; i < 150*3; i += 3) {
//			float m = max(max(a[i], a[i + 1]), a[i + 2]);
//			a[i] = (a[i] == m) ? 1 : 0;
//			a[i+1] = (a[i+1] == m) ? 1 : 0;
//			a[i+2] = (a[i+2] == m) ? 1 : 0;
//		}
//		float count = 0;
//		for (int i = 0; i < 150*3; i++) {
//			count += a[i] * t[i];
//		}
//		float errorRate = (150 - count)/150;
//		System.out.println(count);
		float errorRate = mlp.test(tp1, false);
		assertEquals(0, errorRate, 0.01);

	}

}
