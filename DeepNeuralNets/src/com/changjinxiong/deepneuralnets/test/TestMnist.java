package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.awt.Dimension;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;

public class TestMnist {

	@Test
	public void testMnistDataProvider() {
//        String path = System.getProperty("user.dir"); 
//        System.out.println(path);
		JFrame frame = new JFrame();
		frame.setMinimumSize(new Dimension(384, 384));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
        JLabel imageLabel = new JLabel();
        JLabel type = new JLabel("!!!!");
		frame.getContentPane().add(imageLabel);
		frame.getContentPane().add(type);
		imageLabel.setBounds(128, 80, 128, 128);
		type.setBounds(192, 250, 120, 16);
		imageLabel.setVisible(true);
		type.setVisible(true);
		frame.getContentPane().setVisible(true);
		frame.setVisible(true);
//		MnistDataProvider mp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", 1, false);
		MnistDataProvider mp = new MnistDataProvider("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", 1, false);
		int width = 28;
		int height = 28;
		
		for (int i = 0; i < mp.getDatasetSize(); i ++) {
			float[] floatPixels = mp.getNextbatchInput();
			int[] pixels = new int[floatPixels.length] ;
			for (int j = 0; j < pixels.length; j++) {
				pixels[j ] = (int) (floatPixels[j] * 255);
			}
			
			BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
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
            
            float[] label = mp.getNextBatchLabel();
			for (int j = 0; j < 10; j++) {
				if (label[j] == 1){
					type.setText(""+j);
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
	public void train() {
		boolean useOpenCL = true;
		boolean addBias = true;
		int batchSize = 100;
		MnistDataProvider mnistTraining = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		MnistDataProvider mnistTest = new MnistDataProvider("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", 10000, false);
		int[] mlpLayers = new int[]{784,300,10};
		NeuralNetwork mlp = new MultiLayerPerceptron(mlpLayers, addBias, useOpenCL); 
//		mlp.test(mnistTraining);
//		mlp.test(mnistTest);
		int costType = 0; //cross entropy
		int epoch = 1;
		float baselearningRate = 0.5f;
		float momentum = 0.9f;
		float weightDecay = 0;
		int lrChangeCycle = mnistTraining.getDatasetSize() / mnistTraining.getBatchSize();
		float lrChangeRate = 0.8f;
		Logger logger = Logger.getLogger("MNIST traing with MLP");
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
		logger.log(Level.INFO, "Training start...");
		mlp.train(mnistTraining, costType, baselearningRate, momentum, weightDecay, lrChangeCycle, lrChangeRate, epoch);
//		mlp.test(mnistTraining);
//		System.out.println("Test with test set...");
		float errorRate = mlp.test(mnistTest);
		assertEquals(0, errorRate, 0.07);
		
		logger.log(Level.INFO, "Saving weights...");
		String path = "/home/jxchang/project/records/mnist/.mlp.weights";
		mlp.saveWeights(path);
		logger.log(Level.INFO, "Weights saved to " + path);

//		NeuralNetwork mlp1 = new MultiLayerPerceptron(new int[]{784,300,10}, true, useOpenCL); 
//		mlp1.loadWeights(path + "/.MultiLayerPerceptron.weights");
//		mnistTest.reset();
//		mlp1.test(mnistTest);

	}

	
	@Test
	public void demo() {
		JFrame frame = new JFrame();
		frame.setMinimumSize(new Dimension(384, 384));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
        JLabel imageLabel = new JLabel();
        JLabel type = new JLabel("");
        JLabel pred = new JLabel("");
		frame.getContentPane().add(imageLabel);
		frame.getContentPane().add(type);
		frame.getContentPane().add(pred);
		imageLabel.setBounds(128, 80, 128, 128);
		type.setBounds(128, 250, 120, 16);
		pred.setBounds(192, 250, 120, 16);
		imageLabel.setVisible(true);
		type.setVisible(true);
		pred.setVisible(true);
		frame.getContentPane().setVisible(true);
		frame.setVisible(true);
		int width = 28;
		int height = 28;
		
		
		boolean useOpenCL = false;
		boolean addBias = true;
		
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{784,300,10}, addBias, useOpenCL); 
		mlp.loadWeights("/home/jxchang/project/records/mnist/mlp784_300_10_bias.weights");
		
		MnistDataProvider mp = new MnistDataProvider("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", 1, false);
		float[] image, l;
		for (int i = 0; i < mp.getDatasetSize(); i ++) {
			image = mp.getNextbatchInput();
			mlp.fordwardPass(image);
			float[] a = mlp.getOutputLayer().getActivations();
			int prediction = 0;
			for (int j = 0; j < mp.getLabelDimension() - 1; j++) {
				if (a[prediction] > a[j + 1]) {
					a[j + 1] = 0f;
				} else {
					a[prediction] = 0f;
					prediction = j + 1;
				}
			}
			a[prediction] = 1f;
			int label = 0;
			l = mp.getNextBatchLabel();
			for (int j = 0; j < 10 ; j++) {
				if (l[j] == 1) {
					label = j;
					break;
				}
			}
		
			int[] pixels = new int[image.length ];
			for (int j = 0; j < pixels.length; j++) {
				pixels[j ] = (int) (image[j] * 255);
			}
			
			BufferedImage BImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
			Image images = BImage.getScaledInstance(128, 128, Image.SCALE_SMOOTH);
            WritableRaster raster =  BImage.getRaster();
            raster.setPixels(0,0,width,height,pixels);

            imageLabel.setIcon(new ImageIcon(images));
            type.setText("label: "+ label);
            pred.setText("prediction: "+ prediction);

//			System.out.printf("%d %s \n%s \n", pixels.length,Arrays.toString(pixels),Arrays.toString(label));           
    		try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
	}
	
}
