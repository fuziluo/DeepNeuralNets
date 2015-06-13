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

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.junit.Test;

import com.changjinxiong.deepneuralnets.nn.MultiLayerPerceptron;
import com.changjinxiong.deepneuralnets.nn.NeuralNetwork;
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
			float[] floatPixels = cp.getNextbatchInput(false);
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
	public void Test() {
		boolean useOpenCL = true;
		boolean addBias = true;
		int batchSize = 128;
		int costType = 0; //cross entropy
		CIFAR10DataProvider trainingSet = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", batchSize, DatasetType.TRAINING_ALL, false);
		CIFAR10DataProvider TestSet = new CIFAR10DataProvider("/home/jxchang/project/datasets/CIFAR/cifar-10-batches-bin", 10000, DatasetType.TEST, false);
		
		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{3072, 1024, 512 ,10}, addBias, useOpenCL); 
//		mlp.test(trainingSet);
		mlp.test(TestSet);
		System.out.println("Training start...");
		int decayCycle = 20 * trainingSet.getDatasetSize()/trainingSet.getBatchSize();
		mlp.train(trainingSet, costType, 0.03f, 0.9f, 0, decayCycle, 0.33f, 80);
		mlp.test(trainingSet);
//		System.out.println("Test with test set...");
		float errorRate = mlp.test(TestSet);
		assertEquals(0, errorRate, 0.1);		
	}
	

}
