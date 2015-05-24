package com.changjinxiong.deepneuralnets.nn;
/**
 * 
 * @author jxchang
 *
 */
public interface Layer {
	public float[] getWeight();
	public void updateWeights(float learningRate, float momentum);
	public void setWeight(float[] weights);	
	public void backpropagation(boolean useOpenCL); //training
	public float[] getActivations();
	public void setInputs(float[] inputs);
	public void forwardPass(boolean useOpenCL); //forward pass
	public Layer getPreviousLayer();
//	public void setPreviousLayer(Layer previousLayer);
	public Layer getNextLayer();
	public void setNextLayer(Layer nextLayer);
	public float[] getErrors();
	public float[] getPrevErrors();
	public void setErrors(float[] error);
	public float[] getGradients();
	public int getBatchSize();
	public boolean hasBias();
	public int getNumOfNodes();
}
