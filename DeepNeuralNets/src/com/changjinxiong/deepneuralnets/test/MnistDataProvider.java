package com.changjinxiong.deepneuralnets.test;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class MnistDataProvider implements DataProvider {
    private RandomAccessFile images;
    private RandomAccessFile labels;
    private int rows;
    private int cols;
    private int datasetSize;
    private ArrayList<Integer> indexSeq;
    private int currentIndex;
    private int batchSize;
	private boolean random;
//    private float[] currentImages;
    private float[] currentLabelsBatch;    
	public MnistDataProvider(String imagesFile, String labelsFile, int batchSize, boolean random) {
		try {
		    this.images = new RandomAccessFile(imagesFile, "r");
		    this.labels = new RandomAccessFile(labelsFile, "r");

		    // magic numbers
		    images.readInt();
		    datasetSize = images.readInt();
		    rows = images.readInt();
		    cols = images.readInt();
		    currentLabelsBatch = new float[10 * batchSize];
			if (batchSize < 1 || batchSize > datasetSize) {
				throw new IllegalArgumentException("batchSize must be withi 1 and " + datasetSize);
			}
			this.batchSize = batchSize;
			indexSeq = new ArrayList<Integer>(datasetSize);
			for (int i = 0; i < datasetSize; i++) {
				indexSeq.add(i);
			}
			this.random = random;
			if (random) {
				Collections.shuffle(indexSeq, new Random()); //fix seed here for debugging TODO
			}
		} catch (IOException e) {
		    e.printStackTrace();
		}
	}

	@Override
	public float[] getNextbatchInput() {
		int dataSize = rows * cols;
		int dataSizeWithBias = dataSize ;
	    byte[] currentImage = new byte[rows * cols];
//	    byte[] currentLabel = new byte[10]; 
		float[] result = new float[dataSizeWithBias * batchSize];
		currentLabelsBatch = new float[10 * batchSize]; //10 kinds of digits
		for (int i = 0; i < batchSize; i++) {
			//image data
		    try {
				images.seek(16 + dataSize * indexSeq.get(currentIndex));
				images.readFully(currentImage);
			    for (int j = 0; j < dataSize; j++) {
			    	result[i * dataSizeWithBias + j] = (currentImage[j] & 0xFF)/255.0f;
			    }
			} catch (IOException e) {
				e.printStackTrace();
			}
//			if (bias) { //bias activation is always 1
//				result[i * dataSizeWithBias + dataSize] = 1;
//			}
			//label data
			try {
			    labels.seek(8 + currentIndex);
			    currentLabelsBatch[i * 10 + labels.readUnsignedByte()] = 1;
			} catch (IOException e) {
			    e.printStackTrace();
			}
			currentIndex ++;
			if (currentIndex >= datasetSize) {
				if (random) {
					Collections.shuffle(indexSeq, new Random());
				}
				currentIndex = currentIndex % datasetSize;
			}
		}
//		System.out.println(Arrays.toString(result));
		return result;
	}

	@Override
	public float[] getNextBatchLabel() {
		return currentLabelsBatch;
	}

	@Override
	public int getDatasetSize() {
		return datasetSize;
	}

	@Override
	public int getBatchSize() {
		return batchSize;
	}

	@Override
	public void reset() {
		currentIndex = 0;
	}

	@Override
	public int getDataDimemsion() {
		return rows * cols;
	}

	@Override
	public int getLabelDimension() {
		return 10;
	}

}
