package com.changjinxiong.deepneuralnets.test;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class CIFAR10DataProvider implements DataProvider {
	public enum DatasetType {TRAINING_ALL, TEST};
	private RandomAccessFile[] datasets;
    private ArrayList<Integer> indexSeq;
    private int currentIndex;
    private int datasetSize;
    private int batchSize;
    private float[] currentLabelsBatch;  
	private boolean random;
    
	public CIFAR10DataProvider(String path, int batchSize, DatasetType datasetType, boolean random) {
		if (datasetType == DatasetType.TRAINING_ALL) {
			datasets = new RandomAccessFile[5];
			try {
				this.datasets[0] = new RandomAccessFile(Paths.get(path, "data_batch_1.bin").toString(), "r");
				this.datasets[1] = new RandomAccessFile(Paths.get(path, "data_batch_2.bin").toString(), "r");
				this.datasets[2] = new RandomAccessFile(Paths.get(path, "data_batch_3.bin").toString(), "r");
				this.datasets[3] = new RandomAccessFile(Paths.get(path, "data_batch_4.bin").toString(), "r");
				this.datasets[4] = new RandomAccessFile(Paths.get(path, "data_batch_5.bin").toString(), "r");
				this.batchSize = batchSize;
				this.datasetSize = 50000;
				if (batchSize < 1 || batchSize > datasetSize) {
					throw new IllegalArgumentException("batchSize must be withi 1 and " + datasetSize);
				}
				currentLabelsBatch = new float[10 * batchSize];
				indexSeq = new ArrayList<Integer>(datasetSize);
				for (int i = 0; i < datasetSize; i++) {
					indexSeq.add(i);
				}
				this.random = random;
				if (random) {
					Collections.shuffle(indexSeq, new Random()); //fix seed here for debugging TODO
				}			
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		} else if (datasetType == DatasetType.TEST) {
			datasets = new RandomAccessFile[1];
			try {
				this.datasets[0] = new RandomAccessFile(Paths.get(path, "test_batch.bin").toString(), "r");
				this.batchSize = batchSize;
				this.datasetSize = 10000;
				if (batchSize < 1 || batchSize > datasetSize) {
					throw new IllegalArgumentException("batchSize must be withi 1 and " + datasetSize);
				}
				currentLabelsBatch = new float[10 * batchSize];
				indexSeq = new ArrayList<Integer>(datasetSize);
				for (int i = 0; i < datasetSize; i++) {
					indexSeq.add(i);
				}
				this.random = random;
				if (random) {
					Collections.shuffle(indexSeq, new Random()); //fix seed here for debugging TODO
				}			
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		} else {
			throw new IllegalArgumentException("DatasetType not supported yet: " + datasetType);
		}
	}

	@Override
	public float[] getNextbatchInput() {
		int dataSize = 3 * 32 * 32;
		float[] result = new float[dataSize * batchSize];
	    byte[] currentImage = new byte[dataSize];
	    float[] currentLabel = new float[10 * batchSize];
		for (int i = 0; i < batchSize; i++) {
		    int index = indexSeq.get(currentIndex);
			int currentDatasetNo = (index) / 10000;
			int localInd = (index) % 10000;
			try {
				datasets[currentDatasetNo].seek(localInd * (dataSize + 1));
				int label = datasets[currentDatasetNo].readUnsignedByte();
				currentLabel[i * 10 + label] = 1;
				currentLabelsBatch = currentLabel;
				datasets[currentDatasetNo].readFully(currentImage);
				//******subtract image mean********
				float mean1 = 0, mean2 = 0, mean3 = 0;
				for (int j = 0; j < 1024; j++) {
					mean1 += currentImage[j] & 0xFF;
					mean2 += currentImage[1024 + j] & 0xFF;
					mean3 += currentImage[2048 + j] & 0xFF;										
				}
//				for (int j = 0; j < 1024; j++) {
//					currentImage[j] -= mean1 / 1024;
//					currentImage[1024 + j] -= mean2 / 1024;
//					currentImage[2048 + j] -= mean3 / 1024;
//				}
				//***********************************
				for (int j = 0; j < dataSize; j++) {
					if (j < 1024)
						result[i * dataSize + j] = (currentImage[j] & 0xFF) - mean1 / 1024;
					else if (j < 2048)
						result[i * dataSize + j] = (currentImage[j] & 0xFF) - mean2 / 1024;
					else if (j < 3072)
						result[i * dataSize + j] = (currentImage[j] & 0xFF) - mean3 / 1024;
				}
//				for (int j = 0; j < dataSize; j++) {
//					result[i * dataSize + j] = (currentImage[j] & 0xFF);///256.0f;
//				}
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
	public int getDataDimemsion() {
		return 32 * 32;
	}

	@Override
	public int getLabelDimension() {
		return 10;
	}

	@Override
	public void reset() {
		currentIndex = 0;
	}

}
