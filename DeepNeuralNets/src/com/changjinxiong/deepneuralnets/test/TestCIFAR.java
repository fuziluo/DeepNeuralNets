package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.ConvolutionalNeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.Layer;
import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer;
import com.changjinxiong.deepneuralnets.nn.PoolingLayer.PoolingType;
import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.test.CIFAR10DataProvider.DatasetType;

public class TestCIFAR {

	@Test
	public void testCIFAR10DataProvider() {
		JFrame frame = new JFrame();
		frame.setMinimumSize(new Dimension(384, 384));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
        JLabel imageLabel = new JLabel();
        JLabel type = new JLabel("!!!!");
		frame.getContentPane().add(imageLabel);
		frame.getContentPane().add(type);
		imageLabel.setBounds(128, 80, 128, 128);
		type.setBounds(128, 208, 120, 16);
		imageLabel.setVisible(true);
		type.setVisible(true);
		frame.getContentPane().setVisible(true);
		frame.setVisible(true);
		CIFAR10DataProvider cp = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", 1, DatasetType.TEST, false);
		int width = 32;
		int height = 32;
		String[] labels= {"airplane", 
				"automobile",
				"bird",
				"cat",
				"deer",
				"dog",
				"frog",
				"horse",
				"ship", 
				"truck"};
		
		for (int i = 0; i < cp.getDatasetSize(); i ++) {
			float[] floatPixels = cp.getNextbatchInput();
			int[] pixels = new int[floatPixels.length] ;
			for (int j = 0; j < pixels.length/3; j++) {
				pixels[j * 3] = (int) (floatPixels[j] * 255);
				pixels[j * 3 + 1] = (int) (floatPixels[pixels.length/3 + j] * 255);
				pixels[j * 3 + 2] = (int) (floatPixels[2 * pixels.length/3 + j] * 255);
			}
			
			BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
//          try {
//				image = ImageIO.read(new File("/home/jxchang/project/datasets/LFW/lfw-deepfunneled/AJ_Cook/AJ_Cook_0001.jpg"));
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
			Image images = image.getScaledInstance(128, 128, Image.SCALE_SMOOTH);
            WritableRaster raster =  image.getRaster();
            raster.setPixels(0,0,width,height,pixels);

            imageLabel.setIcon(new ImageIcon(images));
            
            float[] label = cp.getNextBatchLabel();
			for (int j = 0; j < 10; j++) {
				if (label[j] == 1){
					type.setText(""+labels[j]);
					break;
				}
			}  

//			System.out.printf("%d %s \n%s \n", pixels.length,Arrays.toString(pixels),Arrays.toString(label));           
    		try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
//    		while(true);
		}
	}

	@Test
	public void TrainMLP() {
		boolean useOpenCL = true;
		boolean addBias = true;
		int batchSize = 64;
		CIFAR10DataProvider trainingSet = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);
		CIFAR10DataProvider TestSet = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", 10000, DatasetType.TEST, false);
		int costType = 0; //cross entropy
		float baselearningRate = 0.02f;
		float momentum = 0.9f;
		float weightDecay = 0.05f;
		int lrChangeCycle = 0;//10 * trainingSet.getDatasetSize()/trainingSet.getBatchSize();
		float lrChangeRate = 0.33f;
		int epoch = 20;
		int[] mlpLayers = new int[]{3072, 1024, 512 ,10};
		NeuralNetwork mlp = new MultiLayerPerceptron(mlpLayers, addBias, useOpenCL); 
		
		Logger logger = Logger.getLogger("CIFAR10 traing with MLP");
		logger.log(Level.INFO, "MLP architecture: \n"
				+ "{0} {1} bias \n", new Object[] {Arrays.toString(mlpLayers), addBias ? "with" : "without"});

		logger.log(Level.INFO, "Traning configuration: \n"
				+ "useOpenCL = {0} \n"
				+ "batchSize = {1} \n"
				+ "costType = {2} \n"
				+ "epoch = {3} \n"
				+ "baselearningRate = {4} \n"
				+ "momentum = {5} \n"
				+ "weightDecay = {6} \n"
				+ "lrChangeCycle = {7} \n"
				+ "lrChangeRate = {8} \n", new Object[] {useOpenCL, batchSize, (costType==0?"CE":"MSE"), epoch, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate});
		
//		mlp.test(trainingSet);
		mlp.test(TestSet);
		
		logger.log(Level.INFO, "Training start...");
		mlp.train(trainingSet, costType, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate, epoch);

		logger.log(Level.INFO, "Saving weights...");
		String path = "/home/jxchang/project/records/cifar/.mlp.weights";
		mlp.saveWeights(path);
		logger.log(Level.INFO, "Weights saved to " + path);

		mlp.test(trainingSet);
//		System.out.println("Test with test set...");
		float errorRate = mlp.test(TestSet);
		assertEquals(0, errorRate, 0.1);		
	}
	

	@Test
	public void TrainCNN() {
		boolean useOpenCL = true;
		boolean addBias = true;
		boolean padding = true;
		int batchSize = 100;
		CIFAR10DataProvider trainingSet = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);
		CIFAR10DataProvider TestSet = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TEST, false);
		int costType = 1; 
		float baselearningRate = 0.003f;
		float momentum = 0.9f;
		float weightDecay = 0;//0.01f;
		int lrChangeCycle = 5 * trainingSet.getDatasetSize()/trainingSet.getBatchSize();
		float lrChangeRate = 0.33f;
		int epoch = 10;
		int[][] cnnLayers = new int[][] {	{3, 0, 0 ,0}, 
											{32, 5, 5, 1},
											{2, 2, 2}, 
											{32, 5, 5, 1},
											{2, 2, 2}, 
											{64, 5, 5, 1}, 
											{2, 2, 2}, 
											{64},
											{10}
											};
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(cnnLayers, addBias, padding, useOpenCL); 
		cnn.setInputShape(new int[] {32, 32});
		Layer l1 = cnn.getInputLayer();
		Layer l2 = l1.getNextLayer();
		PoolingLayer l3 = (PoolingLayer) l2.getNextLayer();
		Layer l4 = l3.getNextLayer();
		PoolingLayer l5 = (PoolingLayer) l4.getNextLayer();
		Layer l6 = l5.getNextLayer();
		PoolingLayer l7 = (PoolingLayer) l6.getNextLayer();
		Layer l8 = l7.getNextLayer();
		Layer l9 = l8.getNextLayer();
		l2.setActivationType(ActivationType.TANH);
		l4.setActivationType(ActivationType.TANH);
		l6.setActivationType(ActivationType.TANH);
		l8.setActivationType(ActivationType.TANH);
		l9.setActivationType(ActivationType.TANH);
		l3.setPoolingType(PoolingType.MAX);
		l5.setPoolingType(PoolingType.AVER);
		l7.setPoolingType(PoolingType.AVER);
		
		Logger logger = Logger.getLogger("CIFAR10 traing with CNN");
		logger.log(Level.INFO, "CNN architecture: \n"
				+ "{0} {1} bias \n"
				+ "conv layer activation type: {2}\n"
				+ "fully layer activation type: {3}"
				, new Object[] {Arrays.deepToString(cnnLayers), addBias ? "with" : "without", ActivationType.TANH, ActivationType.TANH});

		logger.log(Level.INFO, "Traning configuration: \n"
				+ "useOpenCL = {0} \n"
				+ "batchSize = {1} \n"
				+ "costType = {2} \n"
				+ "epoch = {3} \n"
				+ "baselearningRate = {4} \n"
				+ "momentum = {5} \n"
				+ "weightDecay = {6} \n"
				+ "lrChangeCycle = {7} \n"
				+ "lrChangeRate = {8} \n", new Object[] {useOpenCL, batchSize, (costType==0?"CE":"MSE"), epoch, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate});

		
//		System.out.println(Arrays.toString(l2.getWeight()));
		logger.log(Level.INFO, "Pretest before training...");
		cnn.test(TestSet);
		
		String path = "/home/jxchang/project/records/cifar/.cnn.weights";
		cnn.loadWeights(path);

		cnn.test(TestSet);
		
		
		cnn.train(trainingSet, costType, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate, epoch);
//		System.out.println(Arrays.toString(l2.getWeight()));


		cnn.test(trainingSet);
//		System.out.println("Test with test set...");
		
		cnn.saveWeights(path);

		float errorRate = cnn.test(TestSet);
		assertEquals(0, errorRate, 0.1);	
		
		logger.log(Level.INFO, "Saving weights...");


	}
	
}
