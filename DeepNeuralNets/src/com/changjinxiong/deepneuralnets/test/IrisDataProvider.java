package com.changjinxiong.deepneuralnets.test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
/**
 * 
 * @author jxchang
 *
 */
public class IrisDataProvider implements DataProvider{
	/**
	 * Iris dataset (http://archive.ics.uci.edu/ml/datasets/Iris)
	 * the first 50 samplea are class 1
	 * the following samples are class 2
	 * the last 50 are class 3
	 */
    private static final float[][] irisData = {
	{ 5.1f,3.5f,1.4f,0.2f },
	{ 4.9f,3.0f,1.4f,0.2f },
	{ 4.7f,3.2f,1.3f,0.2f },
	{ 4.6f,3.1f,1.5f,0.2f },
	{ 5.0f,3.6f,1.4f,0.2f },
	{ 5.4f,3.9f,1.7f,0.4f },
	{ 4.6f,3.4f,1.4f,0.3f },
	{ 5.0f,3.4f,1.5f,0.2f },
	{ 4.4f,2.9f,1.4f,0.2f },
	{ 4.9f,3.1f,1.5f,0.1f },
	{ 5.4f,3.7f,1.5f,0.2f },
	{ 4.8f,3.4f,1.6f,0.2f },
	{ 4.8f,3.0f,1.4f,0.1f },
	{ 4.3f,3.0f,1.1f,0.1f },
	{ 5.8f,4.0f,1.2f,0.2f },
	{ 5.7f,4.4f,1.5f,0.4f },
	{ 5.4f,3.9f,1.3f,0.4f },
	{ 5.1f,3.5f,1.4f,0.3f },
	{ 5.7f,3.8f,1.7f,0.3f },
	{ 5.1f,3.8f,1.5f,0.3f },
	{ 5.4f,3.4f,1.7f,0.2f },
	{ 5.1f,3.7f,1.5f,0.4f },
	{ 4.6f,3.6f,1.0f,0.2f },
	{ 5.1f,3.3f,1.7f,0.5f },
	{ 4.8f,3.4f,1.9f,0.2f },
	{ 5.0f,3.0f,1.6f,0.2f },
	{ 5.0f,3.4f,1.6f,0.4f },
	{ 5.2f,3.5f,1.5f,0.2f },
	{ 5.2f,3.4f,1.4f,0.2f },
	{ 4.7f,3.2f,1.6f,0.2f },
	{ 4.8f,3.1f,1.6f,0.2f },
	{ 5.4f,3.4f,1.5f,0.4f },
	{ 5.2f,4.1f,1.5f,0.1f },
	{ 5.5f,4.2f,1.4f,0.2f },
	{ 4.9f,3.1f,1.5f,0.1f },
	{ 5.0f,3.2f,1.2f,0.2f },
	{ 5.5f,3.5f,1.3f,0.2f },
	{ 4.9f,3.1f,1.5f,0.1f },
	{ 4.4f,3.0f,1.3f,0.2f },
	{ 5.1f,3.4f,1.5f,0.2f },
	{ 5.0f,3.5f,1.3f,0.3f },
	{ 4.5f,2.3f,1.3f,0.3f },
	{ 4.4f,3.2f,1.3f,0.2f },
	{ 5.0f,3.5f,1.6f,0.6f },
	{ 5.1f,3.8f,1.9f,0.4f },
	{ 4.8f,3.0f,1.4f,0.3f },
	{ 5.1f,3.8f,1.6f,0.2f },
	{ 4.6f,3.2f,1.4f,0.2f },
	{ 5.3f,3.7f,1.5f,0.2f },
	{ 5.0f,3.3f,1.4f,0.2f },
	{ 7.0f,3.2f,4.7f,1.4f },
	{ 6.4f,3.2f,4.5f,1.5f },
	{ 6.9f,3.1f,4.9f,1.5f },
	{ 5.5f,2.3f,4.0f,1.3f },
	{ 6.5f,2.8f,4.6f,1.5f },
	{ 5.7f,2.8f,4.5f,1.3f },
	{ 6.3f,3.3f,4.7f,1.6f },
	{ 4.9f,2.4f,3.3f,1.0f },
	{ 6.6f,2.9f,4.6f,1.3f },
	{ 5.2f,2.7f,3.9f,1.4f },
	{ 5.0f,2.0f,3.5f,1.0f },
	{ 5.9f,3.0f,4.2f,1.5f },
	{ 6.0f,2.2f,4.0f,1.0f },
	{ 6.1f,2.9f,4.7f,1.4f },
	{ 5.6f,2.9f,3.6f,1.3f },
	{ 6.7f,3.1f,4.4f,1.4f },
	{ 5.6f,3.0f,4.5f,1.5f },
	{ 5.8f,2.7f,4.1f,1.0f },
	{ 6.2f,2.2f,4.5f,1.5f },
	{ 5.6f,2.5f,3.9f,1.1f },
	{ 5.9f,3.2f,4.8f,1.8f },
	{ 6.1f,2.8f,4.0f,1.3f },
	{ 6.3f,2.5f,4.9f,1.5f },
	{ 6.1f,2.8f,4.7f,1.2f },
	{ 6.4f,2.9f,4.3f,1.3f },
	{ 6.6f,3.0f,4.4f,1.4f },
	{ 6.8f,2.8f,4.8f,1.4f },
	{ 6.7f,3.0f,5.0f,1.7f },
	{ 6.0f,2.9f,4.5f,1.5f },
	{ 5.7f,2.6f,3.5f,1.0f },
	{ 5.5f,2.4f,3.8f,1.1f },
	{ 5.5f,2.4f,3.7f,1.0f },
	{ 5.8f,2.7f,3.9f,1.2f },
	{ 6.0f,2.7f,5.1f,1.6f },
	{ 5.4f,3.0f,4.5f,1.5f },
	{ 6.0f,3.4f,4.5f,1.6f },
	{ 6.7f,3.1f,4.7f,1.5f },
	{ 6.3f,2.3f,4.4f,1.3f },
	{ 5.6f,3.0f,4.1f,1.3f },
	{ 5.5f,2.5f,4.0f,1.3f },
	{ 5.5f,2.6f,4.4f,1.2f },
	{ 6.1f,3.0f,4.6f,1.4f },
	{ 5.8f,2.6f,4.0f,1.2f },
	{ 5.0f,2.3f,3.3f,1.0f },
	{ 5.6f,2.7f,4.2f,1.3f },
	{ 5.7f,3.0f,4.2f,1.2f },
	{ 5.7f,2.9f,4.2f,1.3f },
	{ 6.2f,2.9f,4.3f,1.3f },
	{ 5.1f,2.5f,3.0f,1.1f },
	{ 5.7f,2.8f,4.1f,1.3f },
	{ 6.3f,3.3f,6.0f,2.5f },
	{ 5.8f,2.7f,5.1f,1.9f },
	{ 7.1f,3.0f,5.9f,2.1f },
	{ 6.3f,2.9f,5.6f,1.8f },
	{ 6.5f,3.0f,5.8f,2.2f },
	{ 7.6f,3.0f,6.6f,2.1f },
	{ 4.9f,2.5f,4.5f,1.7f },
	{ 7.3f,2.9f,6.3f,1.8f },
	{ 6.7f,2.5f,5.8f,1.8f },
	{ 7.2f,3.6f,6.1f,2.5f },
	{ 6.5f,3.2f,5.1f,2.0f },
	{ 6.4f,2.7f,5.3f,1.9f },
	{ 6.8f,3.0f,5.5f,2.1f },
	{ 5.7f,2.5f,5.0f,2.0f },
	{ 5.8f,2.8f,5.1f,2.4f },
	{ 6.4f,3.2f,5.3f,2.3f },
	{ 6.5f,3.0f,5.5f,1.8f },
	{ 7.7f,3.8f,6.7f,2.2f },
	{ 7.7f,2.6f,6.9f,2.3f },
	{ 6.0f,2.2f,5.0f,1.5f },
	{ 6.9f,3.2f,5.7f,2.3f },
	{ 5.6f,2.8f,4.9f,2.0f },
	{ 7.7f,2.8f,6.7f,2.0f },
	{ 6.3f,2.7f,4.9f,1.8f },
	{ 6.7f,3.3f,5.7f,2.1f },
	{ 7.2f,3.2f,6.0f,1.8f },
	{ 6.2f,2.8f,4.8f,1.8f },
	{ 6.1f,3.0f,4.9f,1.8f },
	{ 6.4f,2.8f,5.6f,2.1f },
	{ 7.2f,3.0f,5.8f,1.6f },
	{ 7.4f,2.8f,6.1f,1.9f },
	{ 7.9f,3.8f,6.4f,2.0f },
	{ 6.4f,2.8f,5.6f,2.2f },
	{ 6.3f,2.8f,5.1f,1.5f },
	{ 6.1f,2.6f,5.6f,1.4f },
	{ 7.7f,3.0f,6.1f,2.3f },
	{ 6.3f,3.4f,5.6f,2.4f },
	{ 6.4f,3.1f,5.5f,1.8f },
	{ 6.0f,3.0f,4.8f,1.8f },
	{ 6.9f,3.1f,5.4f,2.1f },
	{ 6.7f,3.1f,5.6f,2.4f },
	{ 6.9f,3.1f,5.1f,2.3f },
	{ 5.8f,2.7f,5.1f,1.9f },
	{ 6.8f,3.2f,5.9f,2.3f },
	{ 6.7f,3.3f,5.7f,2.5f },
	{ 6.7f,3.0f,5.2f,2.3f },
	{ 6.3f,2.5f,5.0f,1.9f },
	{ 6.5f,3.0f,5.2f,2.0f },
	{ 6.2f,3.4f,5.4f,2.3f },
	{ 5.9f,3.0f,5.1f,1.8f },
    };
    private final int datasetSize = irisData.length;
    private ArrayList<Integer> indexSeq = new ArrayList<Integer>(datasetSize);
    private int currentIndex;
    private int batchSize;
	private boolean random;
	private float[] labels;
	public IrisDataProvider(int batchSize, boolean random) {
		currentIndex = 0;
		if (batchSize < 1 || batchSize > datasetSize) {
			throw new IllegalArgumentException("batchSize must be withi 1 and " + datasetSize);
		}
		this.batchSize = batchSize;
		for (int i = 0; i < datasetSize; i++) {
			indexSeq.add(i);
		}
		this.random = random;
		if (random) {
			Collections.shuffle(indexSeq, new Random()); //fix seed here for debugging TODO
		}
	}

	public float[] getNextbatchInput() {
		int dataSize = irisData[0].length;
		int dataSizeWithBias = dataSize;
		float[] result = new float[dataSizeWithBias * batchSize];
		labels = new float[3 * batchSize]; //3 kinds of iris
		for (int i = 0; i < batchSize; i++) {
			System.arraycopy(irisData[indexSeq.get(currentIndex)], 0, result, i * dataSizeWithBias, dataSize);
//			if (bias) { //bias activation is always 1
//				result[i * dataSizeWithBias + dataSize] = 1;
//			}
			if (indexSeq.get(currentIndex) < 50) {
				labels[i * 3] = 1;
			} else if (indexSeq.get(currentIndex) < 100) {
				labels[i * 3 + 1] = 1;
			} else {
				labels[i * 3 + 2] = 1;
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
	public float[] getNextBatchLabel() {
		return labels;
	}
	public void reset() {
		currentIndex = 0;
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
		return 4;
	}

	@Override
	public int getLabelDimension() {
		// TODO Auto-generated method stub
		return 3;
	}
}
