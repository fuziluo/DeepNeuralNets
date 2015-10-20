package com.changjinxiong.deepneuralnets.nn;

public interface FeatureMapLayer extends Layer {
	/**
	 * Simply return the caculated outputFeatureMapsShapes
	 * @return outputFeatureMapsShapes
	 */
	public int[] getOutputFeatureMapsShapes();
	/**
	 * 
	 * @return the number of feature maps
	 */
	public int getNumOfFeatureMaps();
	/**
	 * Used to set the shape of input feature maps. This method also calculates 
	 * the shape of output feature maps
	 * @param inputShape, a 2d array
	 */
	public void setInputShape(int[] inputShape);
	/**
	 * 
	 * @param padding true means padding will be added to the boundaries of the feature maps
	 */
	public void setPadding(boolean padding);
}
