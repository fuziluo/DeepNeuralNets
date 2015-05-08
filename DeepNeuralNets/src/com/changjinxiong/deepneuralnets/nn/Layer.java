package com.changjinxiong.deepneuralnets.nn;
/**
 * 
 * @author jxchang
 *
 */
public interface Layer {
	public float[] getWeight();
	public void updateWeights(float learningRate);
//	public void setWeight(float[] weights);	
	public void backpropagation(boolean useOpenCL); //training
	public float[] getActivations();
	public void setActivations(float[] activations);
	public void forwardPass(boolean useOpenCL); //forward pass
	public Layer getPreviousLayer();
//	public void setPreviousLayer(Layer previousLayer);
	public Layer getNextLayer();
	public void setNextLayer(Layer nextLayer);
	public float[] getError();
	public float[] getGradients();
	public void setError(float[] error);
	public int getBatchSize();
	public void setBatchSize(int batchSize);
	public boolean hasBias();
	public int getNumOfNodes();
}
