package com.changjinxiong.deepneuralnets.test;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.ByteChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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
	private boolean zeroMean;
	private boolean normalization;
	private float[] mean = new float[3072];
	private DatasetType datasetType;
    
	public CIFAR10DataProvider(String path, int batchSize, DatasetType datasetType, boolean random) {
		this(path, batchSize, datasetType, random, true, false);
	}
	
	public CIFAR10DataProvider(String path, int batchSize, DatasetType datasetType, boolean random, boolean zeroMean, boolean normalization) {
		this.datasetType = datasetType;
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
				this.zeroMean = zeroMean;
				this.normalization = normalization;
				if (random) {
					Collections.shuffle(indexSeq, new Random()); //fix seed here for debugging TODO
				}	
				//********
				computeMean(path);
				//********
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
				this.zeroMean = zeroMean;
				this.normalization = normalization;
				if (random) {
					Collections.shuffle(indexSeq, new Random()); //fix seed here for debugging TODO
				}	
				//********
				computeMean(path);
				//********
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		} else {
			throw new IllegalArgumentException("DatasetType not supported yet: " + datasetType);
		}
	}

	private void computeMean(String p) {
		String path = Paths.get(p, "mean.mean").toString();
		File f = new File(path);
		if(f.exists() && !f.isDirectory()) { 
			try {
				ByteChannel channel = Files.newByteChannel(Paths.get(path), READ);
				int bufferSize = 4 * 3072;
				ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
				buffer.clear();
				channel.read(buffer);
				buffer.flip();
				buffer.asFloatBuffer().get(mean);
				channel.close();	
//				System.out.println(Arrays.toString(mean));
			} catch (IOException e) {
				e.printStackTrace();
			}
			return; 
		}
		
		if (datasetType != DatasetType.TRAINING_ALL) {
			throw new IllegalStateException("no mean file found, do traing first.");
		}
		
		try {
			for (int k = 0; k < 3072; k++) {
				for (int i = 0; i < 5; i++) {
					for (int j = 0; j < 10000; j++) {
						datasets[i].seek(j * 3073 + 1 + k);
						mean[k] += datasets[i].readUnsignedByte()& 0xFF;
					}
				}
				mean[k] /= 50000;
//				System.out.println(k + "  " + mean[k]);
			}
			ByteChannel channel = Files.newByteChannel(Paths.get(path), WRITE, CREATE, TRUNCATE_EXISTING);
			int bufferSize = 4 * 3072;
			ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
//			buffer.asFloatBuffer().put(mean);
			channel.write(buffer);
			channel.close();	
			System.out.println("mean saved to " + path);
		} catch (IOException e) {
			e.printStackTrace();
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
//				//******subtract image mean********
//				float mean = 0;
//				for (int j = 0; j < 3072; j++) {
//					mean += currentImage[j] & 0xFF;
//				}
//				//***********************************
//				for (int j = 0; j < dataSize; j++) {
//					if (j < 1024)
//						result[i * dataSize + j] = (currentImage[j] & 0xFF) - mean / 3072;
//					else if (j < 2048)
//						result[i * dataSize + j] = (currentImage[j] & 0xFF) - mean / 3072;
//					else if (j < 3072)
//						result[i * dataSize + j] = (currentImage[j] & 0xFF) - mean / 3072;
//				}
				for (int j = 0; j < dataSize; j++) {
//					result[i * dataSize + j] = ((currentImage[j] & 0xFF) - (zeroMean ? (mean / 3072) : 0)) / (normalization ? 255 : 1);
					result[i * dataSize + j] = ((currentImage[j] & 0xFF) - (zeroMean ? (mean[j]) : 0)) / (normalization ? 255 : 1);
				}
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
