package com.changjinxiong.deepneuralnets.test;

import static org.junit.Assert.*;

import java.awt.Dimension;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.Arrays;

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
		MnistDataProvider mp = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", 1, false);
		int width = 28;
		int height = 28;
		
		for (int i = 0; i < mp.getDatasetSize(); i ++) {
			float[] floatPixels = mp.getNextbatchInput(false);
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
	public void test() {
		boolean useOpenCL = true;
		int batchSize = 100;
		int costType = 0; //cross entropy
		MnistDataProvider mnistTraining = new MnistDataProvider("test/train-images-idx3-ubyte", "test/train-labels-idx1-ubyte", batchSize, false);
		MnistDataProvider mnistTest = new MnistDataProvider("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", 10000, false);

		NeuralNetwork mlp = new MultiLayerPerceptron(new int[]{784,300,10}, true, useOpenCL); 
//		mlp.test(mnistTraining);
//		mlp.test(mnistTest);
		System.out.println("Training start...");
		mlp.train(mnistTraining, costType, 0.5f, 0.9f, 0, mnistTraining.getDatasetSize() / mnistTraining.getBatchSize(), 0.8f, 1);
//		mlp.test(mnistTraining);
//		System.out.println("Test with test set...");
		float errorRate = mlp.test(mnistTest);
		assertEquals(0, errorRate, 0.05);

	}
	
}
