package com.changjinxiong.deepneuralnets.test;
import static java.lang.Math.*;
import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.IrisDataProvider;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;

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
		IrisDataProvider tp = new IrisDataProvider(1, true);
		for (int i = 0; i < 150; i ++) {
			
			System.out.printf("%d %s %s",i+1,Arrays.toString(tp.getNextbatchInput(false)),Arrays.toString(tp.getNextBatchLabel()));
			System.out.println();

		}

	}
	@Test
	public void test() {
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(new int[]{4,20,20,3}, true); //overfitting
		int batchSize = 150;
		IrisDataProvider tp = new IrisDataProvider(batchSize, false);
		mlp.train(tp, 0.005f, 30000);
		//test
		mlp.fordwardPass(tp.getNextbatchInput(true));
		float[] a = mlp.getOutputLayer().getActivations();
		float[] t = tp.getNextBatchLabel();

		for (int i = 0; i < batchSize*3; i += 3) {
			float m = max(max(a[i], a[i + 1]), a[i + 2]);
			a[i] = (a[i] == m) ? 1 : 0;
			a[i+1] = (a[i+1] == m) ? 1 : 0;
			a[i+2] = (a[i+2] == m) ? 1 : 0;
		}
//		System.out.println(Arrays.toString(a));
//		System.out.println(Arrays.toString(t));
		float count = 0;
		for (int i = 0; i < batchSize*3; i++) {
			count += a[i] * t[i];
		}
		float errorRate = (150 - count)/150;
		System.out.println(count);
		assertEquals(0, errorRate, 0.03);

//		fail("Not yet implemented");
	}

}
