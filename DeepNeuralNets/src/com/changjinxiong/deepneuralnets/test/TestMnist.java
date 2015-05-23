package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;

public class TestMnist {

	@Test
	public void testMnistDataProvider() {
//        String path = System.getProperty("user.dir"); 
//        System.out.println(path);
		MnistDataProvider mp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", 1, false);
		for (int i = 0; i < 2; i ++) {
			System.out.printf("%d %s %s",i+1,Arrays.toString(mp.getNextbatchInput(true)),Arrays.toString(mp.getNextBatchLabel()));
			System.out.println();
		}
//		fail("Not yet implemented");
	}

	@Test
	public void test() {
		boolean useOpenCL = true;
		int batchSize = 100;
		MnistDataProvider mnistTraining = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		MnistDataProvider mnistTest = new MnistDataProvider("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", 10000, false);

		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{784,300,10}, true); 
//		mlp.test(mnistTraining, useOpenCL);
//		mlp.test(mnistTest, useOpenCL);
		mlp.train(mnistTraining, 0.005f, 0.9f, mnistTraining.getDatasetSize() / mnistTraining.getBatchSize(), 0.8f, 1, useOpenCL);
//		mlp.test(mnistTraining, useOpenCL);
		float errorRate = mlp.test(mnistTest, useOpenCL);
		assertEquals(0, errorRate, 0.05);


	}
	
}
