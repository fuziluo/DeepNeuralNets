package com.changjinxiong.deepneuralnets.test;
/**
 * 
 * @author jxchang
 *
 */
public interface DataProvider {
	public float[] getNextbatchInput();
	public float[] getNextBatchLabel();
	public int getDatasetSize();
	public int getBatchSize();
	public int getDataDimemsion();
	public int getLabelDimension();
	public void reset();
}
