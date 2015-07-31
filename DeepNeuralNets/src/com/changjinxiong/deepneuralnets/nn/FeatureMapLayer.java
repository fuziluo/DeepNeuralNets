package com.changjinxiong.deepneuralnets.nn;

public interface FeatureMapLayer extends Layer {
	public int[] getOutputFeatureMapsShapes();
//	public void updateFeatureMapsShapes();
	public int getNumOfFeatureMaps();
	public void setInputShape(int[] inputShape);
	public void setPadding(boolean padding);
}
