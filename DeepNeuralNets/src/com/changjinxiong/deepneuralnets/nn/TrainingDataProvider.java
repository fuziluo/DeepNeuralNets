package com.changjinxiong.deepneuralnets.nn;
/**
 * 
 * @author jxchang
 *
 */
public interface TrainingDataProvider {
	public float[] getNextbatchInput(boolean bias);
	public float[] getNextBatchLabel();
	public int getDatasetSize();
	public int getBatchSize();
	public void reset();
}
